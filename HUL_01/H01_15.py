import tensorflow as tf

# Set random seed to generate reproducible results
# When using random is very important take care of set.seed (global) and particular seed
tf.random.set_seed(1)

# Use normal distribution draws to generate tensors
A = tf.random.normal([200])
B = tf.random.normal([200])

# Perform dot product (that is a scalar)
c = tf.tensordot(A,B, axes = 1)

# Print numpy argument of c
print(c.numpy())
