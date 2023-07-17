import tensorflow as tf
from tensorflow import keras

# Solve an OLS model with tf.estimator()


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

# Define feature columns
features = [
    tf.feature_column.numeric_column('constant'),
    tf.feature_column.numeric_column('x1')
]

# Define model
ols = tf.estimator.LinearRegressor(features)

# Define function to feed data to model
def train_input_fn():
    features = {'constant': [1, 1], 'x1': [0,2]}
    target = [2, 4]
    return features, target

# Train OLS model
ols.train(train_input_fn, steps = 100)

# Define feature columns
def test_input_fn():
    features = {'constant': [1,1], 'x1': [3,5]}
    return features

# Define prediction generator
predict_gen = ols.predict(input_fn=test_input_fn)

# Generate predictions
predictions = [next(predict_gen) for j in range(2)]

# Print predictions
print(predictions)
