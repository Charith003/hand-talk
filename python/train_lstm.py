"""Train a 3-layer stacked LSTM on collected keypoint sequences.

Expects dataset/<class>/<sample_idx>/sequence.npy from collect_data.py.
Saves best_model.keras and labels.npy.
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization, Input,
)
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
)
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

DATA_PATH = "dataset"
SEQ_LENGTH = 30
FEATURES = 126

CLASSES = sorted(
    d for d in os.listdir(DATA_PATH)
    if os.path.isdir(os.path.join(DATA_PATH, d))
)

sequences, labels = [], []
for cls in CLASSES:
    cls_dir = os.path.join(DATA_PATH, cls)
    for sample in os.listdir(cls_dir):
        f = os.path.join(cls_dir, sample, "sequence.npy")
        if os.path.exists(f):
            seq = np.load(f)
            if seq.shape == (SEQ_LENGTH, FEATURES):
                sequences.append(seq)
                labels.append(cls)

X = np.array(sequences, dtype=np.float32)
le = LabelEncoder()
y = le.fit_transform(labels)
y_cat = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=y,
)
print(f"X_train {X_train.shape} | classes {list(le.classes_)}")

model = Sequential([
    Input(shape=(SEQ_LENGTH, FEATURES)),
    LSTM(128, return_sequences=True),
    BatchNormalization(),
    Dropout(0.3),
    LSTM(256, return_sequences=True),
    BatchNormalization(),
    Dropout(0.3),
    LSTM(128),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation="relu"),
    Dropout(0.2),
    Dense(len(le.classes_), activation="softmax"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

callbacks = [
    ModelCheckpoint("best_model.keras", save_best_only=True, monitor="val_accuracy"),
    EarlyStopping(patience=15, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=7, min_lr=1e-6),
]

model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100, batch_size=32, callbacks=callbacks,
)

loss, acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {acc:.4f}")
np.save("labels.npy", le.classes_)