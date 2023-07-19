import tensorflow as tf

# Define x as a constant
x = tf.constant(2.0)

# Define f(g(x)) within an instance of gradient tape
with tf.GradientTape() as t:
    t.watch(x)
    y = x**3
    f = 5*y**2

# Compute gradient of f with respect to x
df_dx = t.gradiet(f,x)
print (df_dx.numpy())
