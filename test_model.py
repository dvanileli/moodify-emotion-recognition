import tensorflow as tf

model = tf.keras.models.load_model("emotion_model.h5")

print("Model berhasil dimuat!")
print("Input Shape:", model.input_shape)
print("Output Shape:", model.output_shape)