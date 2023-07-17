import tensorflow as tf

# Use normal distribution draws to generate tensors
A = tf.random.normal([200,50])
B = tf.random.normal([50,10])

# Perform matrix multiplication
C = tf.matmul(A,B)

# Print shape of C
print(C.shape)

