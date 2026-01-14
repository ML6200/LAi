#!/usr/bin/env python3
import struct
import os

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

    def save(self, path: str):
        """Save vocabulary to binary file (matching C++ format)"""
        print(f"Saving vocabulary to {path}...")
        print(f"Vocab size: {len(self.id_to_char)}")
        
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

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    tokenizer = SimpleTokenizer()
    tokenizer.save("data/vocab.bin")
    print("Done.")
