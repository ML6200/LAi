#!/usr/bin/env python3
"""
LAi Training Script
Train a small transformer model for Hungarian + English

Usage:
    python train.py --config mini --epochs 10 --batch_size 32
"""

import os
import math
import json
import struct
import argparse
from dataclasses import dataclass
from typing import Optional, List, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint


@dataclass
class ModelConfig:
    """Model configuration matching C++ implementation"""
    dim: int = 512
    hidden_dim: int = 2048
    n_layers: int = 12
    n_heads: int = 8
    n_kv_heads: int = 8
    vocab_size: int = 32000
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5

    @staticmethod
    def mini():
        return ModelConfig(dim=512, hidden_dim=2048, n_layers=12, n_heads=8, vocab_size=32000, max_seq_len=1024)

    @staticmethod
    def tiny():
        return ModelConfig(dim=384, hidden_dim=1536, n_layers=8, n_heads=6, vocab_size=32000, max_seq_len=512)

    @staticmethod
    def small():
        return ModelConfig(dim=768, hidden_dim=3072, n_layers=16, n_heads=12, vocab_size=32000, max_seq_len=2048)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0):
    """Precompute rotary embeddings"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """Apply rotary embeddings to Q and K
    xq, xk: [batch, seq_len, n_heads, head_dim]
    freqs_cis: [seq_len, head_dim//2] complex
    """
    # Reshape to complex: [batch, seq_len, n_heads, head_dim//2]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # freqs_cis is [seq_len, head_dim//2], need to broadcast to [1, seq_len, 1, head_dim//2]
    freqs_cis = freqs_cis[:xq_.shape[1]]  # Trim to actual seq_len
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim//2]

    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    """Multi-head attention with RoPE"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.dim // config.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor] = None):
        bsz, seqlen, _ = x.shape

        xq = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = self.wk(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        # RoPE
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # Expand KV heads for GQA
        if self.n_rep > 1:
            xk = xk.repeat_interleave(self.n_rep, dim=2)
            xv = xv.repeat_interleave(self.n_rep, dim=2)

        # Attention
        xq = xq.transpose(1, 2)  # (bsz, n_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Efficient attention (FlashAttention)
        if hasattr(F, 'scaled_dot_product_attention'):
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                attn_mask=mask,
                is_causal=True if mask is None else False
            )
        else:
            # Fallback for older PyTorch
            scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores + mask
            attn = F.softmax(scores, dim=-1)
            output = torch.matmul(attn, xv)

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    """SwiGLU FFN"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.w_gate = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.w_up = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.w_down = nn.Linear(config.hidden_dim, config.dim, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class TransformerBlock(nn.Module):
    """Transformer layer"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor] = None):
        x = x + self.attention(self.attention_norm(x), freqs_cis, mask)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class Transformer(nn.Module):
    """Full transformer model"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.dim, config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Tie embeddings
        self.output.weight = self.tok_embeddings.weight

        # Precompute RoPE frequencies
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(config.dim // config.n_heads, config.max_seq_len, config.rope_theta)
        )

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None):
        bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)

        # Mask is handled implicitly by is_causal=True in Attention
        mask = None 

        freqs_cis = self.freqs_cis[:seqlen]

        for layer in self.layers:
            # Gradient checkpointing
            if self.training:
                h = checkpoint(layer, h, freqs_cis, mask, use_reentrant=False)
            else:
                h = layer(h, freqs_cis, mask)

        h = self.norm(h)

        if targets is not None:
            # Memory optimization: Compute loss in chunks to avoid materializing full logits
            # Full logits: [B, T, V] -> 32*1024*32000 * 2 bytes = ~2GB
            # Chunked: Process small segments at a time
            logits = None # We don't return full logits during training to save memory
            loss = 0.0
            chunk_size = 64 # Small chunk size
            
            # Flatten for easier chunking
            h_flat = h.view(-1, self.config.dim)
            targets_flat = targets.view(-1)
            
            for i in range(0, h_flat.size(0), chunk_size):
                end = min(i + chunk_size, h_flat.size(0))
                h_chunk = h_flat[i:end]
                t_chunk = targets_flat[i:end]
                
                # Compute logits only for this chunk
                logits_chunk = self.output(h_chunk)
                
                # Compute loss for this chunk
                loss_chunk = F.cross_entropy(logits_chunk, t_chunk, ignore_index=-100, reduction='sum')
                loss += loss_chunk
            
            # Average the loss
            # Note: We should technically divide by (total_tokens - ignored_tokens), 
            # but usually dividing by total_tokens is close enough or we can count.
            # Here we divide by batch_size * seq_len for simplicity and consistency with standard mean reduction
            loss = loss / (bsz * seqlen)
            
        else:
            # Inference mode: only compute last token logits usually, but here we do all
            logits = self.output(h)
            loss = None

        return logits, loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


class TextDataset(Dataset):
    """Simple text dataset for training"""
    def __init__(self, texts: List[str], tokenizer, max_len: int = 512):
        self.samples = []
        for text in texts:
            tokens = tokenizer.encode(text)
            # Split into chunks
            for i in range(0, len(tokens) - max_len, max_len // 2):
                chunk = tokens[i:i + max_len + 1]
                if len(chunk) > 10:
                    self.samples.append(chunk)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens = self.samples[idx]
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y


def collate_fn(batch):
    """Pad batch to same length"""
    max_len = max(len(x) for x, y in batch)
    xs = torch.full((len(batch), max_len), 0, dtype=torch.long)
    ys = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for i, (x, y) in enumerate(batch):
        xs[i, :len(x)] = x
        ys[i, :len(y)] = y

    return xs, ys


class SimpleTokenizer:
    """Simple character-level tokenizer for bootstrapping"""
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.char_to_id = {}
        self.id_to_char = {}

        # Special tokens
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.unk_id = 3

        # Initialize with basic chars
        special = ['<pad>', '<bos>', '<eos>', '<unk>']
        for i, s in enumerate(special):
            self.char_to_id[s] = i
            self.id_to_char[i] = s

        # Add ASCII + common Unicode
        idx = len(special)
        for c in range(256):
            char = chr(c)
            if char not in self.char_to_id:
                self.char_to_id[char] = idx
                self.id_to_char[idx] = char
                idx += 1

        # Hungarian specific characters
        hungarian_chars = 'áéíóöőúüűÁÉÍÓÖŐÚÜŰ'
        for c in hungarian_chars:
            if c not in self.char_to_id:
                self.char_to_id[c] = idx
                self.id_to_char[idx] = c
                idx += 1

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        tokens = []
        if add_bos:
            tokens.append(self.bos_id)
        for c in text:
            tokens.append(self.char_to_id.get(c, self.unk_id))
        if add_eos:
            tokens.append(self.eos_id)
        return tokens

    def decode(self, tokens: List[int]) -> str:
        chars = []
        for t in tokens:
            if t in [self.pad_id, self.bos_id, self.eos_id]:
                continue
            chars.append(self.id_to_char.get(t, '?'))
        return ''.join(chars)

    def save(self, path: str):
        """Save vocabulary to binary file (matching C++ format)"""
        with open(path, 'wb') as f:
            f.write(struct.pack('I', 0x4C414956))  # Magic: "LAIV"
            f.write(struct.pack('I', 1))  # Version
            f.write(struct.pack('I', len(self.id_to_char)))  # Vocab size

            for i in range(len(self.id_to_char)):
                token = self.id_to_char[i]
                token_bytes = token.encode('utf-8')
                f.write(struct.pack('I', len(token_bytes)))
                f.write(token_bytes)
                f.write(struct.pack('f', 0.0))  # Score


def export_model(model: Transformer, path: str):
    """Export model to C++ binary format"""
    config = model.config

    with open(path, 'wb') as f:
        # Magic and version
        f.write(b'LAi1')
        f.write(struct.pack('I', 1))

        # Config (must match C++ struct layout)
        f.write(struct.pack('i', config.dim))
        f.write(struct.pack('i', config.hidden_dim))
        f.write(struct.pack('i', config.n_layers))
        f.write(struct.pack('i', config.n_heads))
        f.write(struct.pack('i', config.n_kv_heads))
        f.write(struct.pack('i', config.vocab_size))
        f.write(struct.pack('i', config.max_seq_len))
        f.write(struct.pack('f', config.rope_theta))
        f.write(struct.pack('f', config.norm_eps))
        f.write(struct.pack('i', 0))  # activation type

        # Weight dtype
        f.write(struct.pack('B', 0))  # F32

        # Offsets (placeholder)
        vocab_offset_pos = f.tell()
        f.write(struct.pack('Q', 0))  # vocab_offset
        weights_offset_pos = f.tell()
        f.write(struct.pack('Q', 0))  # weights_offset

        # Padding to 256 bytes for alignment
        header_size = f.tell()
        if header_size < 256:
            f.write(b'\x00' * (256 - header_size))

        # Record weights offset
        weights_offset = f.tell()
        f.seek(weights_offset_pos)
        f.write(struct.pack('Q', weights_offset))
        f.seek(weights_offset)

        # Write weights in order expected by C++
        def write_tensor(tensor):
            data = tensor.detach().cpu().float().numpy()
            f.write(data.tobytes())

        # Embeddings
        write_tensor(model.tok_embeddings.weight)
        write_tensor(model.output.weight)
        write_tensor(model.norm.weight)

        # Layers
        for layer in model.layers:
            write_tensor(layer.attention.wq.weight)
            write_tensor(layer.attention.wk.weight)
            write_tensor(layer.attention.wv.weight)
            write_tensor(layer.attention.wo.weight)
            write_tensor(layer.feed_forward.w_gate.weight)
            write_tensor(layer.feed_forward.w_up.weight)
            write_tensor(layer.feed_forward.w_down.weight)
            write_tensor(layer.attention_norm.weight)
            write_tensor(layer.ffn_norm.weight)

    print(f"Model exported to {path}")
    print(f"  Config: {config.dim}d, {config.n_layers}L, {config.n_heads}H")
    print(f"  Parameters: {model.num_params() / 1e6:.1f}M")
    print(f"  File size: {os.path.getsize(path) / 1e6:.1f} MB")


def train(config: ModelConfig, train_texts: List[str], epochs: int = 10,
          batch_size: int = 4, lr: float = 3e-4, device: str = "cuda"):
    """Train the model"""
    print(f"Training LAi model:")
    print(f"  Config: {config.dim}d, {config.n_layers}L, {config.n_heads}H")
    print(f"  Device: {device}")
    print(f"  Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")

    if device == "cuda" and torch.cuda.is_available():
        free_mem, total_mem = torch.cuda.mem_get_info()
        print(f"  GPU Memory: {free_mem/1024**3:.2f}GB free / {total_mem/1024**3:.2f}GB total")
        if free_mem < 2 * 1024**3:
            print("  WARNING: Low GPU memory! Consider restarting runtime or reducing batch size further.")

    # Initialize
    tokenizer = SimpleTokenizer(config.vocab_size)
    model = Transformer(config).to(device)
    print(f"  Parameters: {model.num_params() / 1e6:.1f}M")

    # Dataset and dataloader
    dataset = TextDataset(train_texts, tokenizer, config.max_seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                           collate_fn=collate_fn, num_workers=2, pin_memory=True)
    print(f"  Training samples: {len(dataset)}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)

    # Learning rate scheduler
    warmup_steps = 100
    total_steps = len(dataloader) * epochs
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixed precision
    scaler = GradScaler('cuda')

    # Training loop
    model.train()
    global_step = 0

    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            with autocast('cuda'):
                _, loss = model(x, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            global_step += 1

            if batch_idx % 100 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Step {batch_idx}/{len(dataloader)}, "
                      f"Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} complete. Average loss: {avg_loss:.4f}")

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Train LAi model")
    parser.add_argument("--config", choices=["tiny", "mini", "small"], default="mini")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--data", type=str, help="Path to training data (text file)")
    parser.add_argument("--output", type=str, default="models/lai-mini.bin")
    parser.add_argument("--vocab_output", type=str, default="data/vocab.bin")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Get config
    if args.config == "tiny":
        config = ModelConfig.tiny()
    elif args.config == "small":
        config = ModelConfig.small()
    else:
        config = ModelConfig.mini()

    # Load training data
    if args.data and os.path.exists(args.data):
        with open(args.data, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        # Sample data for testing
        print("No training data provided. Using sample data for testing.")
        texts = [
            "Hello, how are you? I am fine, thank you!",
            "Szia! Hogy vagy? Köszönöm, jól vagyok.",
            "The quick brown fox jumps over the lazy dog.",
            "A gyors barna róka átugrik a lusta kutya felett.",
            "Machine learning is a subset of artificial intelligence.",
            "A gépi tanulás a mesterséges intelligencia része.",
            "Budapest is the capital of Hungary.",
            "Budapest Magyarország fővárosa.",
        ] * 1000  # Repeat for more training data

    # Train
    model, tokenizer = train(config, texts, args.epochs, args.batch_size, args.lr, args.device)

    # Export
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(args.vocab_output) or '.', exist_ok=True)

    export_model(model, args.output)
    tokenizer.save(args.vocab_output)

    print(f"\nTraining complete!")
    print(f"  Model saved to: {args.output}")
    print(f"  Vocab saved to: {args.vocab_output}")


if __name__ == "__main__":
    main()
