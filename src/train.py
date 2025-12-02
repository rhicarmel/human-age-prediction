import os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


def train_model(model, train_gen, val_gen, save_dir="checkpoints", epochs=20):
    """
    Train the model with checkpoints and early stopping.
    """

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    checkpoint_path = os.path.join(save_dir, "best_model.h5")

    callbacks = [
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_mae",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            patience=5,
            monitor="val_mae",
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    return history
