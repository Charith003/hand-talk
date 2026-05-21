"""Convert best_model.keras → TF.js (float16 quantized) + labels.json.

Copy the output into the React app:
    tfjs_model_quantized/*  →  public/model/
    labels.json             →  public/labels.json
"""
import json
import numpy as np
import tensorflow as tf
import tensorflowjs as tfjs

model = tf.keras.models.load_model("best_model.keras")
tfjs.converters.save_keras_model(
    model,
    "tfjs_model_quantized",
    quantization_dtype_map={"float16": "*"},
)

labels = np.load("labels.npy", allow_pickle=True).tolist()
with open("labels.json", "w") as f:
    json.dump(labels, f)
print(f"Exported {len(labels)} classes.")