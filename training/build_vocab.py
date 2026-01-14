#!/usr/bin/env python3
"""
Build proper BPE vocabulary for LAi
Uses byte-pair encoding to create 32K subword tokens
"""

import struct
import argparse
from collections import defaultdict, Counter
from typing import List, Dict, Tuple
import re


class BPEVocabBuilder:
    """Build BPE vocabulary from text corpus"""

    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.vocab = []
        self.scores = []

    def train(self, texts: List[str], min_freq: int = 2):
        """Train BPE on text corpus"""
        print(f"Training BPE vocabulary (target size: {self.vocab_size})...")

        # Initialize with special tokens
        self.vocab = ['<pad>', '<bos>', '<eos>', '<unk>']
        self.scores = [0.0, 0.0, 0.0, 0.0]

        # Add byte tokens (256 tokens for raw bytes)
        for i in range(256):
            self.vocab.append(chr(i))
            self.scores.append(-100.0)  # Low priority

        print(f"  Added {len(self.vocab)} base tokens (special + bytes)")

        # Count UTF-8 character frequencies
        char_freq = Counter()
        for text in texts:
            i = 0
            while i < len(text):
                char_len = self._utf8_char_len(ord(text[i]))
                char = text[i:i+char_len]
                char_freq[char] += 1
                i += char_len

        # Add frequent characters
        for char, freq in char_freq.items():
            if freq >= min_freq and char not in self.vocab:
                self.vocab.append(char)
                self.scores.append(float(freq))

        print(f"  Added {len(self.vocab) - 260} frequent characters")

        # Tokenize texts into characters
        tokenized_texts = []
        for text in texts:
            chars = []
            i = 0
            while i < len(text):
                char_len = self._utf8_char_len(ord(text[i]))
                chars.append(text[i:i+char_len])
                i += char_len
            tokenized_texts.append(chars)

        # BPE iterations
        iteration = 0
        max_token_len = 20  # Limit token length to prevent whole-line tokens
        while len(self.vocab) < self.vocab_size:
            # Count pair frequencies
            pair_freq = Counter()
            for tokens in tokenized_texts:
                for i in range(len(tokens) - 1):
                    # Skip if resulting token would be too long
                    if len(tokens[i]) + len(tokens[i + 1]) > max_token_len:
                        continue
                    pair = (tokens[i], tokens[i + 1])
                    pair_freq[pair] += 1

            if not pair_freq:
                break

            # Find most frequent pair
            best_pair, best_freq = pair_freq.most_common(1)[0]

            if best_freq < min_freq:
                break

            # Create new token
            new_token = best_pair[0] + best_pair[1]
            score = float(best_freq)

            self.vocab.append(new_token)
            self.scores.append(score)

            # Merge in all texts
            for tokens in tokenized_texts:
                i = 0
                while i < len(tokens) - 1:
                    if tokens[i] == best_pair[0] and tokens[i + 1] == best_pair[1]:
                        tokens[i] = new_token
                        tokens.pop(i + 1)
                    else:
                        i += 1

            iteration += 1
            if iteration % 500 == 0:
                print(f"  Iteration {iteration}: vocab_size={len(self.vocab)}, "
                      f"best_pair={repr(new_token[:20])}, freq={best_freq}")

        print(f"  Final vocabulary size: {len(self.vocab)}")

    def save(self, path: str):
        """Save vocabulary to binary file (C++ compatible format)"""
        print(f"Saving vocabulary to {path}...")

        with open(path, 'wb') as f:
            # Header
            f.write(struct.pack('I', 0x4C414956))  # Magic: "LAIV"
            f.write(struct.pack('I', 1))            # Version
            f.write(struct.pack('I', len(self.vocab)))  # Vocab size

            # Tokens
            for token, score in zip(self.vocab, self.scores):
                token_bytes = token.encode('utf-8')
                f.write(struct.pack('I', len(token_bytes)))
                f.write(token_bytes)
                f.write(struct.pack('f', score))

        print(f"  Saved {len(self.vocab)} tokens")

    @staticmethod
    def _utf8_char_len(first_byte: int) -> int:
        """Get UTF-8 character length from first byte"""
        if (first_byte & 0x80) == 0:
            return 1
        elif (first_byte & 0xE0) == 0xC0:
            return 2
        elif (first_byte & 0xF0) == 0xE0:
            return 3
        elif (first_byte & 0xF8) == 0xF0:
            return 4
        return 1


def main():
    parser = argparse.ArgumentParser(description="Build BPE vocabulary")
    parser.add_argument("--data", type=str, default="data/train.txt",
                       help="Path to training data")
    parser.add_argument("--vocab_size", type=int, default=32000,
                       help="Target vocabulary size")
    parser.add_argument("--min_freq", type=int, default=2,
                       help="Minimum frequency for tokens")
    parser.add_argument("--output", type=str, default="data/vocab.bin",
                       help="Output vocabulary file")
    args = parser.parse_args()

    # Load training data
    print(f"Loading training data from {args.data}...")
    with open(args.data, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    print(f"  Loaded {len(texts)} lines")

    # Build vocabulary
    builder = BPEVocabBuilder(vocab_size=args.vocab_size)
    builder.train(texts, min_freq=args.min_freq)

    # Save
    builder.save(args.output)
    print("Done!")


if __name__ == "__main__":
    main()
