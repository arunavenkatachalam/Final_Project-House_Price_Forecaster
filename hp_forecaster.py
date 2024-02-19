import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import tensorflow as tf
#import h5py
import pickle

# Load data
df = pd.read_csv(
    r'1_Initial_Preprocessed_Data\initial_preprocessed_hp_train.csv')


# Select columns
columns = ['LotArea', 'BedroomAbvGr', 'FullBath', 'GarageCars', 'YearBuilt', 'SalePrice']
df = df[columns]

# Split data into features and target
X = df.iloc[:, 0:5]
y = df.iloc[:, 5:]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

lr = LinearRegression()
lr.fit(X_train, y_train)


pickle.dump(lr, open('Final_Predictions/hp_forecaster_model.pkl', 'wb'))


# # Build and train a neural network model using Keras
# model = tf.keras.models()
# model.add(tf.keras.layers(10, activation='relu', input_shape=(X_train.shape[1],)))
# model.add(tf.keras.layers(1, activation='linear'))
# model.compile(optimizer='adam', loss='mean_squared_error')
# model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# Save the model to an HDF5 file
# model.save('model.h5')










