import matplotlib.pyplot as plt
import numpy as np


def evaluate_model(model, test_gen):
    """
    Evaluate the model on the test generator.
    """
    loss, mae, mse = model.evaluate(test_gen, verbose=1)
    return {"loss": loss, "mae": mae, "mse": mse}


def plot_predictions(model, test_gen, num_samples=9):
    """
    Plot sample predictions vs true ages.
    """

    images, true_ages = next(test_gen)
    preds = model.predict(images).flatten()

    plt.figure(figsize=(10, 10))

    for i in range(min(num_samples, len(images))):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.title(f"True: {true_ages[i]:.0f}\nPred: {preds[i]:.1f}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
