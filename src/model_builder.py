import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    Dropout,
)
from tensorflow.keras.models import Model


def build_resnet50_model(input_shape=(224, 224, 3), learning_rate=1e-4):
    """
    Build a regression model for age prediction using a pretrained ResNet50 backbone.
    """

    base_model = ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape
    )
    base_model.trainable = False  # freeze base layers

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    predictions = Dense(1, activation="linear")(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="mae",
        metrics=["mae", "mse"]
    )

    return model