import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

dataset = pd.read_csv('cancer.csv')

x = dataset.drop(columns = ["diagnosis(1=m, 0=b)"])
y = dataset["diagnosis(1=m, 0=b)"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Create a sequential model
model = tf.keras.models.Sequential()

# Takes the input data, processes it with 256 neurons, and applies the sigmoid activation (to output between 0 and 1).
model.add(tf.keras.layers.Dense(256, input_shape=x_train.shape[1:], activation = 'sigmoid'))

# Add another hidden layer with 256 neurons.
model.add(tf.keras.layers.Dense(256, activation='sigmoid'))

# Output layer with 1 neuron for binary classification.
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# Compile model with Adam optimizer and binary crossentropy loss, to track accuracy.
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Train the model on training data for 1000 epochs (1000 times).
model.fit(x_train, y_train, epochs = 1000)

# Evaluate the model on test data
model.evaluate(x_test, y_test)
