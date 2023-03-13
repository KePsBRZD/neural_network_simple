import keras
import tensorflow as tf
import numpy as np

input_data = np.array([0.3, 0.7, 0.9]).reshape(-1,1)
output_data = np.array([0.5, 0.9, 1.0]).reshape(-1,1)

#model = tf.keras.Sequential([
#  tf.keras.layers.Dense(1)
#])
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1, activation='linear'))
model.compile(loss='mse', optimizer='sgd')
fit_results = model.fit(x=input_data, y=output_data, epochs=100)

predicted = model.predict([0.7])
print(predicted)

