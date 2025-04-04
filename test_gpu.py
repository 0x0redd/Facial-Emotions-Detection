import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print("GPU Devices:", tf.config.list_physical_devices('GPU'))

# Test GPU computation
if len(tf.config.list_physical_devices('GPU')) > 0:
    print("\nPerforming a test computation on GPU...")
    with tf.device('/GPU:0'):
        a = tf.random.normal([1000, 1000])
        b = tf.random.normal([1000, 1000])
        c = tf.matmul(a, b)
        print("Matrix multiplication result shape:", c.shape)
        print("GPU test successful!")
else:
    print("\nNo GPU available for computation.") 