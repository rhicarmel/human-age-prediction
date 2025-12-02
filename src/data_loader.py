import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_data_generators(
    data_dir,
    img_size=(224, 224),
    batch_size=32,
    val_split=0.2,
):
    """
    Create training and validation generators using ImageDataGenerator
    and flow_from_dataframe with labels from labels.csv.

    Expected structure:

    data/
      ├── labels.csv       # columns: file_name, real_age
      └── final_files/     # image files, names matching file_name
    """

    labels_path = os.path.join(data_dir, "labels.csv")
    images_dir = os.path.join(data_dir, "final_files")

    df = pd.read_csv(labels_path)

    datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=val_split,
        rotation_range=10,
        horizontal_flip=True,
        zoom_range=0.1,
    )

    # Training generator
    train_gen = datagen.flow_from_dataframe(
        dataframe=df,
        directory=images_dir,
        x_col="file_name",
        y_col="real_age",
        target_size=img_size,
        batch_size=batch_size,
        class_mode="raw",       # regression target
        subset="training",
        shuffle=True,
        seed=42,
    )

    # Validation generator
    val_gen = datagen.flow_from_dataframe(
        dataframe=df,
        directory=images_dir,
        x_col="file_name",
        y_col="real_age",
        target_size=img_size,
        batch_size=batch_size,
        class_mode="raw",       # regression target
        subset="validation",
        shuffle=False,
        seed=42,
    )

    return train_gen, val_gen