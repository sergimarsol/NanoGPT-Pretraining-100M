# FineWeb-Edu 10% Sample Dataset Preparation

This script prepares the FineWeb-Edu dataset (10% of the full dataset) for training nanoGPT.

## Setup

1. Run the preparation script:
```bash
python prepare.py
```

This will:
- Download FineWeb-Edu from HuggingFace
- Use only 10% of the training data for efficiency
- Tokenize with GPT-2 BPE (tiktoken)
- Split into 90% train / 10% validation
- Save binary files to `data/fineweb-10pct/`

## Training

After preparation, train with:
```bash
python train.py --dataset=fineweb-10pct --max_iters=50000 --eval_interval=500
```

Or with GPU:
```bash
python train.py --dataset=fineweb-10pct --device=cuda:0 --batch_size=32 --max_iters=50000
```

## Notes

- Download size: ~5GB for initial dataset, reduced to ~500MB for 10%
- Processing time: ~10-15 minutes depending on internet speed and hardware
- The 10% sample is still ~30M tokens, enough for meaningful training
- Output files:
  - `train.bin`: tokenized training data
  - `val.bin`: tokenized validation data
  - `meta.pkl`: vocabulary metadata
