import tensorflow as tf
import numpy as np

tf.random.set_seed(42)

# Create some regression data
X_regression = np.arange(0, 1000, 5).reshape(-1,1)
y_regression = np.arange(100, 1100, 5).reshape(-1,1)

# Split it into training and test sets
X_reg_train = X_regression[:150]
X_reg_test = X_regression[150:]
y_reg_train = y_regression[:150]
y_reg_test = y_regression[150:]

tf.random.set_seed(42)

# Recreate the model
model = tf.keras.Sequential([
  tf.keras.layers.Dense(100),
  tf.keras.layers.Dense(10),
  tf.keras.layers.Dense(1)
])

# Change the loss and metrics of our compiled model
model.compile(loss=tf.keras.losses.mae, # change the loss function to be regression-specific
            optimizer=tf.keras.optimizers.Adam(),
            metrics=['mae']) # change the metric to be regression-specific

# Fit the recompiled model
model.fit(X_reg_train, y_reg_train, epochs=100)

#predicted = model.predict([100])
#print(predicted)