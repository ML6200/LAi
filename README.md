# LAi - Lightweight AI Assistant

A from-scratch efficient LLM assistant written in pure C++ for CPU-only inference on low-end hardware. Optimized for Hungarian and English.

## Features

- **Pure C++ inference** - No Python dependencies for running
- **Extreme efficiency** - Runs on 4GB RAM with CPU-only
- **Bilingual** - Hungarian and English support
- **SIMD optimized** - AVX2 (x86) and NEON (ARM) acceleration
- **Multiple modes** - Chat, Translation, Code, Text processing

## Quick Start

### Build

```bash
# Release build (optimized)
make release

# Debug build (with sanitizers)
make debug

# Run tests
make test

# Run benchmarks
make bench
```

### Run

```bash
# Interactive mode
./lai

# Single prompt
./lai -p "Hello, how are you?"

# Translate English to Hungarian
./lai -t "Hello world"

# Translate Hungarian to English
./lai --to-en "Szia világ"

# Show model info
./lai --info
```

## Training (Google Colab)

1. Install dependencies:
```bash
pip install -r training/requirements.txt
```

2. Prepare training data:
```bash
python training/data.py --output data/train.txt --size small
```

3. Train the model:
```bash
python training/train.py --config mini --epochs 10 --data data/train.txt
```

4. The trained model will be saved to `models/lai-mini.bin`

### Google Colab Notebook

See `training/colab_notebook.ipynb` for a complete training setup that works on free Colab GPUs.

## Model Specifications

### LAi-Mini (Default)
- Parameters: ~150M
- Layers: 12
- Dimension: 512
- Heads: 8
- Context: 1024 tokens
- Memory (Q4): ~75MB
- Memory (F32): ~600MB

### LAi-Tiny (Ultra-low memory)
- Parameters: ~50M
- Layers: 8
- Dimension: 384
- Heads: 6
- Context: 512 tokens
- Memory (Q4): ~25MB

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Memory | <500MB | With Q4 quantization |
| Speed | 15+ tok/s | On modern CPU |
| Startup | <1 sec | Cold start |

## CLI Commands

When running in interactive mode:

| Command | Description |
|---------|-------------|
| `/chat` | Switch to chat mode |
| `/translate` | Switch to translation mode |
| `/code` | Switch to code assistance |
| `/text` | Switch to text processing |
| `/hu <text>` | Translate English → Hungarian |
| `/en <text>` | Translate Hungarian → English |
| `/temp <n>` | Set temperature (0.0-2.0) |
| `/tokens <n>` | Set max tokens |
| `/reset` | Reset conversation context |
| `/stats` | Toggle statistics display |
| `/info` | Show model information |
| `/help` | Show help |
| `/quit` | Exit |

## Project Structure

```
LAi/
├── src/
│   ├── core/           # Tensor ops, SIMD, memory allocator
│   ├── model/          # Transformer architecture
│   ├── tokenizer/      # BPE tokenizer
│   ├── inference/      # Engine, sampler, KV-cache
│   └── cli/            # REPL interface
├── training/           # PyTorch training scripts
├── data/               # Vocabulary, prompts
└── models/             # Trained model weights
```

## Building from Source

### Requirements
- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2019+)
- Make or CMake

### Platform Support
- Linux (x86-64, ARM64)
- macOS (Intel, Apple Silicon)
- Windows (with MinGW or MSVC)

### Build Options

```bash
# Debug with AddressSanitizer
make debug

# Release with LTO
make release

# Check memory leaks
make valgrind

# Run performance benchmarks
make bench
```

## Architecture

The model uses a modern transformer architecture:

- **RMSNorm** - Efficient layer normalization
- **RoPE** - Rotary position embeddings
- **SwiGLU** - Improved FFN activation
- **GQA** - Grouped query attention (optional)

### Memory Optimizations

1. **Q4 Quantization** - 4-bit weights reduce memory 4x
2. **KV-Cache** - Efficient autoregressive generation
3. **Arena Allocator** - Zero-allocation inference
4. **SIMD** - Vectorized operations

## License

MIT License - see LICENSE file

## Contributing

Contributions welcome! Please read CONTRIBUTING.md first.

## Acknowledgments

- Inspired by llama.cpp, but built from scratch
- Hungarian NLP community for language resources
- TinyStories dataset for English training data
