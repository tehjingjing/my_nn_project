# Durian Leaf Disease Classifier

A custom PyTorch CNN for classifying durian leaf images into five disease/health classes. The dataset is included in `data/durian_leaf_dataset/` so the project can be run after cloning. The images come from the Kaggle Durian Leaf Disease Dataset by cthng123; retain this source attribution and the dataset's CC BY 4.0 licence information when redistributing the files.

## 1. Prepare the dataset

Place the extracted dataset folder at `data/durian_leaf_dataset/`. It must contain `train/`, `val/`, and `test/`, each with one subfolder per class:

```text
data/durian_leaf_dataset/
├── train/
│   ├── ALGAL_LEAF_SPOT/
│   ├── ALLOCARIDARA_ATTACK/
│   ├── HEALTHY_LEAF/
│   ├── LEAF_BLIGHT/
│   └── PHOMOPSIS_LEAF_SPOT/
├── val/       # same five class folders
└── test/      # same five class folders
```

If the dataset is stored elsewhere, set `DURIAN_DATA_DIR` to its location, for example `DURIAN_DATA_DIR="/path/to/durian_leaf_dataset" python src/train.py --experiment baseline`. The training and evaluation scripts use `data/durian_leaf_dataset/` by default.

## 2. Train locally

Install the training dependencies, then run the baseline from the project root:

```bash
python -m pip install -r requirements-train.txt
python src/train.py --experiment baseline --epochs 25 --lr 0.001 --batch-size 16 --dropout 0.3
```

Training outputs are saved to `training_runs/baseline/`. To run a controlled experiment, change one setting at a time and use a distinct experiment name. Training uses the training and validation splits; the test split is held out for final evaluation.

## 3. Train with Docker

Build the image from the project root and mount the dataset and model-output folder:

```bash
docker build -f Dockerfile.train -t durian-train .
docker run --rm \
  -v "$(pwd)/data/durian_leaf_dataset:/app/data:ro" \
  -v "$(pwd)/models:/app/models" \
  durian-train --experiment baseline --epochs 25 --lr 0.001 --batch-size 16 --dropout 0.3
```

Docker training outputs are saved in `models/baseline/` on the host computer.

## 4. Evaluate with Docker

Build and run the separate testing image after choosing a checkpoint using validation results:

```bash
docker build -f Dockerfile.test -t durian-test .
docker run --rm \
  -v "$(pwd)/data/durian_leaf_dataset:/app/data:ro" \
  -v "$(pwd)/models:/app/models" \
  -e MODEL_PATH=/app/models/baseline/best_model.pth \
  durian-test
```

Test metrics are written to `models/evaluation/test_results.json`; per-image errors are written to `models/evaluation/wrong_predictions.csv`. The test script also reports the most-common-training-class baseline. Use the held-out test set only for final evaluation after model selection.

## 5. Predict a new image

With the test image built and a checkpoint available, run:

```bash
docker run --rm \
  -v "$(pwd)/models:/app/models:ro" \
  -v "/absolute/path/to/leaf.jpg:/app/leaf.jpg:ro" \
  --entrypoint python durian-test /app/src/predict.py /app/leaf.jpg \
  --model /app/models/baseline/best_model.pth
```

## Reproducibility notes

Random seeds are fixed in the training script. GPU/MPS operations may still vary slightly across devices. The Docker images install the pinned package versions listed in `requirements-train.txt` and `requirements-test.txt`. Keep the checkpoint and record the environment used when comparing results.

## Architecture

The custom CNN has three convolution–ReLU–max-pooling blocks (32, 64 and 128 filters), followed by a 256-unit ReLU dense layer with dropout and a five-logit output. Cross-entropy consumes logits directly; softmax is applied only when reporting probabilities.
