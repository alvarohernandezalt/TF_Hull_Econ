import tensorflow as tf

# Define two scalaras aas constants
s1 = tf.constant(5, tf.float32)
s2 = tf.constant(3, tf.float32)

# Add and multiply using tf.add() and tf.multiply()
s1s2_sum = tf.add(s1,s2)
s1s2_product = tf.multiply(s1, s2)

# Sum and product via overloading operators(outside tensorflow)
### They are the same Tensor result no matter the method used.
ss1s2_sum = s1+s2
ss1s2_product = s1*s2

# Print values
print(s1s2_product,"\n")
print(s1s2_sum,"\n")
print(ss1s2_product,"\n")
print(ss1s2_sum,"\n")