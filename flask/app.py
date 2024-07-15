import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, render_template
from sklearn .preprocessing import MinMaxScaler
app = Flask(__name__, template_folder='templates')
import os

# required definitions
def min_max_scale(test_value, min_val, max_val):
    return (test_value - min_val) / (max_val - min_val)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Loading the historical data for scaling purpose
ds=pd.read_csv('3D printer Material Prediction.csv')

# Define the home route to render the index.html
@app.route('/')
def f():
    return render_template('index.html')

# Define a route to inspect additional information (inspect.html)
@app.route('/inspect')
def inspect():
    return render_template('inspect.html')
# Define a route to predict material type based on input features
@app.route('/predict', methods=['GET', 'POST'])

def predict():

    if request.method == 'POST':
        try:
            # Get input features from the form
            input_features = [x for x in request.form.values()]
            features_value = [np.array(input_features)]
            # Define feature names (assuming the order is fixed)
            features_name = ['layer_height', 'wall_thickness', 'infill_density', 'infill_pattern',
                             'nozzle_temperature', 'bed_temperature', 'print_speed', 'fan_speed',
                             'roughness', 'tension_strenght', 'elongation']
            if len(input_features) != len(features_name):
                raise Exception("Error: Number of input features does not match expected features.")
        except Exception as e:
            print("Oops...!,error in passing input stage :(")
            
        # Reshape input_features to a numpy array
        #input_features=[0.02,8,90,'honey',220,60,40,0,25,18,1.2]
        # converting string to int manually based on input data.
        try:
            input_features1=[]
            for i in range(len(input_features)):
                try:
                    i=float(input_features[i])
                except:
                    if input_features[i].lower()=='grid':
                        i=0
                    else:
                        i=1
                input_features1.append(i)
        except Exception as e:
            print("Oops..!,error in string to float conversion :(")
        
        # Normalizing Input data
        input_scaledd=[]
        for i,j in zip(features_name,input_features1):
            if i!='infill_pattern':
                minn,maxx=ds[i].min(),ds[i].max()
                k=min_max_scale(j,minn,maxx)
                input_scaledd.append(k)
            else:
                k=j
                input_scaledd.append(k)

        print("scaled input features:",input_scaledd)
        
        # Reshape input_features to a numpy array
        input_scaledd_arr = np.array(input_scaledd).reshape(1, -1)
        # Predict using the loaded model
        prediction = model.predict(input_scaledd_arr)
        output = prediction[0]
        # output = 1
        print(output)  # Optional: Print prediction for debugging
        # Render different templates based on prediction
        if (output == 1):
            return render_template('output.html',
                                    prediction_text='The suggested material is ABS. (Acrylonitrile butadiene styrene is a common thermoplastic polymer typically used for injection molding applications)')
        elif output == 0:
            return render_template('output.html',
                                    prediction_text='The suggested material is PLA. (PLA, also known as polyactic acid or polyactide, is a thermoplastic made from renewable resources such as corn starch, tapioca roots, or sugar cane, unlike other industrial materials made primarily from petroleum)')
        else:
            return render_template('output.html',
                                    prediction_text='The given values do not match the range of values of the model. Please ensure the values are within the expected range.')

    return render_template('output.html',
                           prediction_text='The suggested material is ABS. (Acrylonitrile butadiene styrene is a common thermoplastic polymer typically used for injection molding applications)')  # Default return if method is not POST

if __name__ == '__main__':
    app.run( debug=True)
    





