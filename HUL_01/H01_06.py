import tensorflow as tf

# Define the data as constants.
X = tf.constant([[1,0],[1,2]], tf.float32)
Y = tf.constant([[2],[4]], tf.float32)

# Matrix multiply X by X's transpose and invert
beta_0 = tf.linalg.inv(tf.matmul(tf.transpose(X), X))


# Matrix multiply beta_0 by X's transpose
beta_1 = tf.matmul(beta_0, tf.transpose(X))


# Matrix multiply beta_1 by Y
beta = tf.matmul(beta_1, Y)

#Define OLS predict function as static graph
@tf.function
def ols_predict(X, beta):
    y_hat = tf.matmul(X, beta)
    return y_hat

# Predict Y  using X and beta
predictions = ols_predict(X, beta)
