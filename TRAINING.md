# LAi Training Guide

Complete guide for training LAi models from scratch.

## Quick Start (5 minutes)

```bash
# 1. Generate synthetic data
python training/generate_data.py --output data/train.txt \
    --sentences 10000 --stories 5000 --translations 3000 --instructions 2000

# 2. Build vocabulary
python training/build_vocab.py --data data/train.txt \
    --vocab_size 8000 --output data/vocab.bin

# 3. Train model (Apple Silicon)
python training/train.py --config tiny --epochs 10 --batch_size 32 \
    --data data/train.txt --vocab data/vocab.bin \
    --output models/lai-tiny.bin --device mps

# 4. Test
./lai -m models/lai-tiny.bin -p "Hello, how are you?"
```

## Training Pipeline Explained

### Step 1: Data Generation

**Option A: Synthetic Data (Recommended for Testing)**

Fast, no internet required, good for initial testing:

```bash
python training/generate_data.py --output data/train.txt \
    --sentences 10000 \    # Simple HU/EN sentences
    --stories 5000 \       # Short narrative texts
    --translations 3000 \  # Translation pairs
    --instructions 2000    # Task-based examples
```

This creates ~25,000 examples with 65%+ uniqueness.

**Option B: Real Datasets (Better Quality)**

Requires HuggingFace datasets and internet:

```bash
pip install datasets
python training/data.py --output data/train.txt --size small
```

Downloads:
- Hungarian Wikipedia (5,000 articles)
- English TinyStories (20,000 stories)
- Translation pairs from OPUS-100 (5,000 pairs)

### Step 2: Vocabulary Building

Build a BPE (Byte-Pair Encoding) vocabulary:

```bash
python training/build_vocab.py \
    --data data/train.txt \
    --vocab_size 8000 \      # Target vocabulary size
    --min_freq 2 \           # Minimum token frequency
    --output data/vocab.bin
```

**Key Points:**
- Larger vocab = better quality, more memory
- 8K tokens: Good balance for tiny models
- 16K-32K tokens: Better for mini/small models
- The script automatically limits token length to prevent whole-sentence tokens

### Step 3: Model Training

**Device Selection:**

```bash
# Apple Silicon (M1/M2/M3)
python training/train.py --device mps --batch_size 32 ...

# NVIDIA GPU
python training/train.py --device cuda --batch_size 64 ...

# CPU (slower but works everywhere)
python training/train.py --device cpu --batch_size 16 ...
```

**Full Command:**

```bash
python training/train.py \
    --config tiny \              # Model size: tiny, mini, small
    --epochs 10 \                # Training epochs
    --batch_size 32 \            # Batch size (adjust for memory)
    --lr 3e-4 \                  # Learning rate (default is good)
    --data data/train.txt \      # Training data
    --vocab data/vocab.bin \     # Vocabulary file
    --output models/lai-tiny.bin # Output model file
```

**Expected Output:**

```
Loading vocabulary from data/vocab.bin...
  Loaded 8000 tokens
Training LAi model:
  Config: 384d, 8L, 6H
  Device: mps
  Epochs: 10, Batch size: 32, LR: 3.00e-04
  Vocab Size: 8000
  Parameters: 19.4M
  Training samples: 25000
  Epoch 1/10, Step 0/782, Loss: 9.2341, LR: 3.00e-06
  Epoch 1/10, Step 100/782, Loss: 2.1567, LR: 3.00e-04
  ...
  Epoch 10 complete. Average loss: 0.5234

Model exported to models/lai-tiny.bin
  Config: 384d, 8L, 6H
  Vocab size: 8000
  Parameters: 19.4M
  File size: 79.9 MB
```

**Training Progress:**
- Initial loss: ~8-10
- After 1 epoch: ~2-3
- After 10 epochs: <1.0 (good!)
- Time on MPS: ~15 minutes for tiny model

### Step 4: Testing

```bash
# Build C++ inference binary
make release

# Test the model
./lai -m models/lai-tiny.bin -p "Hello"

# Check model info
./lai -m models/lai-tiny.bin --info
```

## Model Configurations

| Config | Params | Dim | Layers | Heads | Context | Memory | Training Time (MPS) |
|--------|--------|-----|--------|-------|---------|--------|---------------------|
| `tiny` | 19M | 384 | 8 | 6 | 512 | 80MB | ~15 min |
| `mini` | 83M | 512 | 12 | 8 | 1024 | 320MB | ~1 hour |
| `small` | 350M | 768 | 16 | 12 | 2048 | 1.4GB | ~4 hours |

## Troubleshooting

### Training Hangs on First Epoch

**Symptom:** Training prints initial info but hangs before showing epoch progress.

**Cause:** MPS + DataLoader multiprocessing conflict.

**Solution:** Already fixed! The script sets `num_workers=0` for MPS.

If you modified the script, ensure:
```python
num_workers = 0 if device == 'mps' else 2
```

### Model Outputs Gibberish

**Symptom:** Model generates random/repetitive text.

**Causes & Solutions:**

1. **Insufficient vocabulary**
   - Check: `python -c "import struct; f=open('data/vocab.bin','rb'); f.seek(8); print(struct.unpack('I',f.read(4))[0])"`
   - Should be: 8000+ tokens
   - If 264: Rebuild vocab with build_vocab.py

2. **Duplicated training data**
   - Check: `grep -v '^$' data/train.txt | sort -u | wc -l`
   - Should be: >60% of total lines
   - If <10%: Use generate_data.py or data.py

3. **Undertrained**
   - Train for more epochs (try 20-30)
   - Check final loss (should be <1.0)

### Out of Memory

**Solutions:**
- Reduce `--batch_size` (try 16 → 8 → 4)
- Use smaller model: `--config tiny`
- Close other applications
- For CPU: Use `--batch_size 4`

### Loss Not Decreasing

**Expected behavior:**
- Epoch 1: Loss ~8-10 → ~2-3
- Epoch 5: Loss ~1-2
- Epoch 10: Loss <1.0

**If loss stays high:**
- Check data quality (run dataset statistics)
- Verify vocab size (should match model config)
- Increase learning rate: `--lr 5e-4`
- Train longer: `--epochs 20`

### Slow Training

**On MPS:**
- Should process ~50-100 batches/second
- If slower: Check Activity Monitor for CPU/GPU usage

**On CPU:**
- Expected: ~5-10 batches/second
- Use smaller batch size to see progress faster

## Dataset Quality Checks

**Check your dataset:**

```bash
# Total lines
wc -l data/train.txt

# Non-empty lines
grep -v '^$' data/train.txt | wc -l

# Unique lines (should be >60% of total)
grep -v '^$' data/train.txt | sort -u | wc -l

# Vocab size
python -c "
import struct
with open('data/vocab.bin', 'rb') as f:
    f.seek(8)
    print('Vocab size:', struct.unpack('I', f.read(4))[0])
"
```

**Good dataset:**
- 20,000+ total examples
- 60%+ unique examples
- 8,000-32,000 token vocabulary
- Mix of tasks (translation, Q&A, generation)

## Advanced: Custom Data

Create your own data generator:

```python
with open('data/custom.txt', 'w') as f:
    # Simple text
    f.write("Your training text here\n")

    # Translation pair
    f.write("<system>Translate English to Hungarian.</system>")
    f.write("<user>Hello</user><assistant>Szia</assistant>\n")

    # Instruction
    f.write("<user>Summarize: Some text here</user>")
    f.write("<assistant>Summary here</assistant>\n")
```

Then build vocab and train as usual.

## Performance Tips

**Faster training:**
- Use MPS/CUDA instead of CPU (10-20× faster)
- Increase batch size (doubles speed, uses more memory)
- Use `--config tiny` during development

**Better quality:**
- Use real datasets (data.py)
- Train for more epochs (20-30)
- Use larger vocab (16K-32K)
- Use larger model (mini/small)

**Memory optimization:**
- Reduce batch size
- Use CPU instead of GPU for smaller models
- Close browser/other apps

## Monitoring Training

Watch for these patterns:

```
✅ Good training:
  Epoch 1/10, Step 0, Loss: 9.23
  Epoch 1/10, Step 100, Loss: 2.45
  Epoch 1/10, Step 200, Loss: 1.87
  ...
  Epoch 10 complete. Average loss: 0.52

❌ Bad training:
  Epoch 1/10, Step 0, Loss: 9.23
  Epoch 1/10, Step 100, Loss: 9.18  # Not decreasing!
  Epoch 1/10, Step 200, Loss: 9.15
```

If loss isn't decreasing, stop and check data quality.
