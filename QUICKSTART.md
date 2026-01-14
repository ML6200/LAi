# LAi Quick Start

## One-Command Training (Copy & Paste)

```bash
# Generate data → Build vocab → Train model (all in one)
python training/generate_data.py --output data/train.txt --sentences 10000 --stories 5000 --translations 3000 --instructions 2000 && \
python training/build_vocab.py --data data/train.txt --vocab_size 8000 --output data/vocab.bin && \
python training/train.py --config tiny --epochs 10 --batch_size 32 --data data/train.txt --vocab data/vocab.bin --output models/lai-tiny.bin --device mps
```

Then test:
```bash
make release && ./lai -m models/lai-tiny.bin -p "Hello, how are you?"
```

## Most Common Commands

### Training

```bash
# 1. Generate training data
python training/generate_data.py --output data/train.txt \
    --sentences 10000 --stories 5000 --translations 3000 --instructions 2000

# 2. Build vocabulary
python training/build_vocab.py --data data/train.txt \
    --vocab_size 8000 --output data/vocab.bin

# 3. Train on Apple Silicon (MPS)
python training/train.py --config tiny --epochs 10 --batch_size 32 \
    --data data/train.txt --vocab data/vocab.bin \
    --output models/lai-tiny.bin --device mps

# 3. Train on NVIDIA GPU
python training/train.py --config tiny --epochs 10 --batch_size 64 \
    --data data/train.txt --vocab data/vocab.bin \
    --output models/lai-tiny.bin --device cuda

# 3. Train on CPU
python training/train.py --config tiny --epochs 10 --batch_size 16 \
    --data data/train.txt --vocab data/vocab.bin \
    --output models/lai-tiny.bin --device cpu
```

### Inference

```bash
# Build C++ binary
make release

# Run tests
./lai --test

# Single prompt
./lai -m models/lai-tiny.bin -p "Hello"

# Interactive mode
./lai -m models/lai-tiny.bin

# Translation
./lai -m models/lai-tiny.bin -t "Good morning"      # EN → HU
./lai -m models/lai-tiny.bin --to-en "Jó reggelt"   # HU → EN

# With parameters
./lai -m models/lai-tiny.bin -p "Tell me a story" --temp 0.9 --tokens 200

# Model info
./lai -m models/lai-tiny.bin --info
```

### Dataset Checks

```bash
# Check dataset size
wc -l data/train.txt

# Check uniqueness (should be >60%)
echo "Total: $(wc -l < data/train.txt)"
echo "Unique: $(grep -v '^$' data/train.txt | sort -u | wc -l)"

# Check vocab size
python -c "import struct; f=open('data/vocab.bin','rb'); f.seek(8); print('Vocab:', struct.unpack('I',f.read(4))[0])"
```

## Device-Specific Settings

| Device | Batch Size | Expected Speed | Notes |
|--------|-----------|----------------|-------|
| Apple M1/M2/M3 | 32-64 | ~100 batch/s | Use `--device mps` |
| NVIDIA GPU | 64-128 | ~200 batch/s | Use `--device cuda` |
| CPU | 8-16 | ~5 batch/s | Use `--device cpu` |

## Training Expectations

| Model | Data | Vocab | Epochs | Time (MPS) | Final Loss |
|-------|------|-------|--------|------------|------------|
| tiny | 25K examples | 8K | 10 | ~15 min | <1.0 |
| mini | 50K examples | 16K | 10 | ~1 hour | <0.5 |
| small | 100K examples | 32K | 10 | ~4 hours | <0.3 |

## Troubleshooting (Quick Fixes)

**Training hangs?**
- Already fixed for MPS! Update your train.py if old version.

**Gibberish output?**
```bash
# Check vocab size (should be 8000, not 264)
python -c "import struct; f=open('data/vocab.bin','rb'); f.seek(8); print(struct.unpack('I',f.read(4))[0])"

# Rebuild if wrong
python training/build_vocab.py --data data/train.txt --vocab_size 8000 --output data/vocab.bin
```

**Out of memory?**
```bash
# Reduce batch size
python training/train.py ... --batch_size 8  # or 4
```

**Loss not decreasing?**
- Check data has >60% unique lines
- Train longer: `--epochs 20`
- Check initial loss is ~8-10 (if much higher, data issue)

## Next Steps

- Read [TRAINING.md](TRAINING.md) for detailed guide
- Read [README.md](README.md) for project overview
- Read [CLAUDE.md](CLAUDE.md) for architecture details
