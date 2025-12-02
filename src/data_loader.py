import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_data_generators(data_dir, img_size=(224, 224), batch_size=32, val_split=0.2):
    """
    Create train and validation generators using ImageDataGenerator.
    """

    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=val_split,
        rotation_range=10,
        horizontal_flip=True,
        zoom_range=0.10,
    )

    train_gen = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="raw",
        subset="training",
        shuffle=True,
    )

    val_gen = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="raw",
        subset="validation",
        shuffle=False,
    )

    return train_gen, val_gen
