import os
import sys

# Ensure the src package is on the path when running from repo root
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from data_loader import create_data_generators
from model_builder import build_resnet50_model
from train import train_model
from evaluate import evaluate_model


def main():
    # Paths
    data_dir = os.path.join(CURRENT_DIR, "data")
    checkpoints_dir = os.path.join(CURRENT_DIR, "checkpoints")

    # Training configuration
    img_size = (224, 224)
    batch_size = 32
    val_split = 0.2
    epochs = 20

    print("Creating data generators...")
    train_gen, val_gen = create_data_generators(
        data_dir=data_dir,
        img_size=img_size,
        batch_size=batch_size,
        val_split=val_split,
    )

    print("Building model...")
    model = build_resnet50_model(input_shape=(*img_size, 3), learning_rate=1e-4)

    print("Starting training...")
    history = train_model(
        model=model,
        train_gen=train_gen,
        val_gen=val_gen,
        save_dir=checkpoints_dir,
        epochs=epochs,
    )

    print("Training complete. Evaluating on validation set...")
    metrics = evaluate_model(model, val_gen)
    print("Validation metrics:", metrics)


if __name__ == "__main__":
    main()