# Groovo-AI

## Create training data

```bash
python src/create_dataset.py \
  --video /path/to/dance.mp4 \
  --song-id hollywood-action
```

The command writes `data/<song-id>.npy`, `data/<song-id>.npz`, and metadata JSON.

## Train the model

```bash
PYTHONPATH=src python src/train_dataset.py \
  --dataset-dir data/references \
  --output-dir artifacts/training
```

All samples from one song stay in the same train, validation, or test split.
