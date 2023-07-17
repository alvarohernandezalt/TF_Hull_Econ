import tensorflow as tf

# Define random 3-tensor of images
images = tf.random.uniform((64, 256, 256))

# Define random 2-tensor image transformation
transform = tf.random.normal((256,256))

# Perform batch matrix multiplication
batch_matmul = tf.matmul(images, transform)

# Perform batch elementwise multiplication
batch_elementwise = tf.multiply(images, transform)

print(batch_matmul.ndim)