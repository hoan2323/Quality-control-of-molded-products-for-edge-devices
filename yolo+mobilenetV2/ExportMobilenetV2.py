import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
import numpy as np
import os

img_size = 224

base_model = MobileNetV2(
    input_shape=(img_size, img_size, 3),
    include_top=False,
    weights='imagenet'
)

model = tf.keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(2, activation='softmax')
])

model.load_weights("best_model.keras")
print("✓ Model loaded")

# Tạo concrete function
@tf.function(input_signature=[tf.TensorSpec(shape=[None, 224, 224, 3], dtype=tf.float32)])
def model_fn(x):
    return model(x, training=False)  # training=False để disable Dropout

# Get concrete function
concrete_func = model_fn.get_concrete_function()

# Convert
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

# Save
output_file = 'mobilenetv2_optimized.tflite'
with open(output_file, 'wb') as f:
    f.write(tflite_model)

# Compare
keras_size = os.path.getsize('best_model.keras') / (1024 * 1024)
tflite_size = os.path.getsize(output_file) / (1024 * 1024)

print(f"\n=== Export thành công ===")
print(f"Keras model: {keras_size:.2f} MB")
print(f"TFLite: {tflite_size:.2f} MB")
print(f"Giảm: {((keras_size - tflite_size) / keras_size * 100):.1f}%")
print(f"\nFile: {output_file}")
