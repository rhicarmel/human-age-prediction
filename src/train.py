import os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


def train_model(
    model,
    train_gen,
    val_gen,
    save_dir="checkpoints",
    epochs=3,
    steps_per_epoch=None,
    validation_steps=None,
):
    """
    Train the model using generators, with optional step limits for quick runs.
    """

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    checkpoint_path = os.path.join(save_dir, "best_model.keras")

    callbacks = [
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_mae",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            patience=3,
            monitor="val_mae",
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    # Limit steps for faster local runs if not specified
    if steps_per_epoch is None:
        steps_per_epoch = min(50, len(train_gen))
    if validation_steps is None:
        validation_steps = min(20, len(val_gen))

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1,
    )

    return history