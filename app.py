#Import librairies
print("Loading Librairies ...")
import pandas as pd
import numpy as np
from math import *
from flask import Flask , render_template , request
import pickle

print("Loading pickled Model ...")
# Load the File and predict unseen data.
with open('static/pickle/xgbmodel.pkl','rb') as model_file:
    pickled_model = pickle.load(model_file)

print("Loading pickled Scaler ...")
# Load the scaler to transform test data
with open('static/pickle/scaler.pkl','rb') as scaler_file:
    scaler = pickle.load(scaler_file)


app = Flask(__name__)

print("Loading : Done")

#Launch the main page
@app.route("/")
def home():
    return render_template('index.html')

#Launch on the button event
@app.route('/', methods=['POST'])
def homepredict():
    pred = [0]*54  #create list of zeros for prediction

    # Update Hr column
    hr = int(request.form['Hour'])
    pred[7+hr]=1

    # update temp 
    temp=float(request.form['Temperature'])
    dtemp=float(request.form['dTemperature'])
    pred[53]=(0.5*temp)+(0.5*dtemp)

    # update humidity
    pred[0]=int(request.form['Humidity'])
    
    # update wndspeed
    pred[1] = float(request.form['WindSpeed'])

    # update visibility
    vis = float(request.form['Visibility'])
    vis_value = 0 if 0<=vis<=399 else (1 if 400<=vis<=999 else 2)
    pred[31+vis_value]=1

    # update solar radiation
    pred[2]=float(request.form['SolarRadiation'])

    # update rainfall
    rain=float(request.form['Rainfall'])
    rain_value = 1 if rain>0 else 0
    pred[3]= rain_value

    # update snowfall
    snow = float(request.form['Snowfall'])
    snow_value = 1 if snow>0 else 0
    pred[4]= snow_value

    # update holiday
    pred[5]=int(request.form['Holiday'])

    # update function day
    pred[6]=int(request.form['FunctioningDay'])

    # update month
    month = int(request.form['Month'])
    pred[33+month]=1

    # update day of the week
    day=int(request.form['Day'])
    pred[45+day]=1
    
    test = scaler.transform(np.array(pred).reshape(1,-1))

    # Testing on one instance
    result = np.square(pickled_model.predict(test))
    return render_template('index.html', prediction = "The prediction for this features is : " + str(ceil(result[0])))

app.run(debug=False)