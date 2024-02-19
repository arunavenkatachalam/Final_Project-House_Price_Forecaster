# Import necessary modules
from flask import Flask, render_template, request, jsonify
import h5py
import numpy as np
import tensorflow as tf
import pickle
import pandas as pd


# Create a Flask application
app = Flask(__name__)

# Linear Regression
model = pickle.load(open('Final_Predictions/hp_forecaster_model.pkl', 'rb'))

# Route wiht less features
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    val1 = request.form['LotArea']
    val2 = request.form['BedroomAbvGr']
    val3 = request.form['FullBath']
    val4 = request.form['GarageCars']
    val5 = request.form['YearBuilt']
    arr = np.array([val1, val2, val3, val4, val5])
    arr = arr.astype(np.float64)
    pred = model.predict([arr])

    return render_template('index.html', data=int(pred))

# Define a route for the home page
@app.route("/")
def index():
    # Render the "index.html" template
    return render_template("index.html")

# Define a route for the House Prices Data
@app.route("/hp_features_data")
def hp_features_data():

    # Read the House Prices data from CSV into a DataFrame
    df= pd.read_csv('3_Preprocessed_Scaled_Data/scaled_preprocessed_hp_data.csv')
    df1=df.to_json(orient='records')
    df1=pd.read_json(df1)

    # Return the House Prices Data to Json
    return jsonify(df1.to_dict(orient='records'))

# Get Predictions through neural network model
# # Load the model from the HDF5 file
# with h5py.File('Final_Predictions/House_Price_Predictions.h5', 'r') as f:
#     model = tf.keras.models.Sequential()
#     # Load model parameters
#     for layer_name in f.keys():
#         # Check if 'units' attribute exists
#         if 'units' in f[layer_name].attrs:
#             units = f[layer_name].attrs['units']
#             layer_config = {'class_name': 'Dense', 'config': {'units': units}}
#             layer = tf.keras.layers.deserialize(layer_config)
#             model.add(layer)
#             for weight_name in f[layer_name].keys():
#                 weight = f[layer_name][weight_name][:]
#                 setattr(layer, weight_name, weight)
#         else:
#             print(f"Skipping layer {layer_name} due to missing 'units' attribute.")

# Function to Predict and show prediction with maximum features
# @app.route('/predict', methods=['POST'])
# def predict():
#     # Extract input values from the form
#     BedroomAbvGr = float(request.form['BedroomAbvGr'])
#     FullBath = float(request.form['FullBath'])
#     HouseStyle = float(request.form['HouseStyle'])
#     FirstFlrSF = float(request.form['1stFlrSF'])
#     YearBuilt = float(request.form['YearBuilt'])
#     TotalBsmtSF = float(request.form['TotalBsmtSF'])
#     BsmtFinSF1 = float(request.form['BsmtFinSF1'])
#     OverallQual_Other = float(request.form['OverallQual_Other'])
#     LotArea = float(request.form['LotArea'])
#     GarageArea = float(request.form['GarageArea'])
#     YearRemodAdd = float(request.form['YearRemodAdd'])
#     TotRmsAbvGrd = float(request.form['TotRmsAbvGrd'])
#     Fireplaces = float(request.form['Fireplaces'])
    
#     # Create a numpy array with the input values
#     arr = np.array([[BedroomAbvGr, FullBath, HouseStyle, FirstFlrSF, YearBuilt, 
#                      TotalBsmtSF, BsmtFinSF1, OverallQual_Other, LotArea, 
#                      GarageArea, YearRemodAdd, TotRmsAbvGrd, Fireplaces]])
    
#     # Make predictions using the loaded model
#     pred = model.predict(arr)
    
#     # Render the template with the prediction
#     return render_template('index.html', prediction=int(pred[0][0]))


# Run the application if this script is executed

if __name__ == "__main__":
    app.run()




