import pickle
import numpy as np
from flask import Flask, jsonify, request
from flask_restful import Api, Resource
import tensorflow as tf 
import joblib

app = Flask(__name__)
api = Api(app)

global model_DL
model_DL = tf.keras.models.load_model('Source/Models/LSTM_Autoencoder.h5')


model_rms = pickle.load(open('Source/Models/lof_rms_trained_model.pkl', 'rb'))
model_mean = pickle.load(open('Source/Models/lof_mean_trained_model.pkl', 'rb'))

def Anomaly_output(x):
    if x==1:
        return "Normal"
    elif x==-1:
        return "Anomaly"
    else:
        return "No Proper Input"

class MakePrediction(Resource):
    def post(self):
        posted_data = request.get_json()
        rms = posted_data["rms"]
        mean = posted_data["mean"]
        if ((rms==0.0) & (mean==0.0)):
            op = 0
        else:
            if rms==0.0:
                op = model_mean.predict(np.array(mean).reshape(-1,1))
            else:
                op = model_rms.predict(np.array(rms).reshape(-1,1))
        Aop = Anomaly_output(op)
        return jsonify({"Output": Aop})

class MakePrediction1(Resource):
    def post(self):
        posted_data1 = request.get_json()
        b1 = posted_data1["Bearing1"]
        b2 = posted_data1["Bearing2"]
        b3 = posted_data1["Bearing3"]
        b4 = posted_data1["Bearing4"]
        b_comb = np.array([b1,b2,b3,b4]).reshape(1,4)
        if ((b1==0.0) & (b2==0.0) & (b3==0.0) & (b4==0.0)):
            op = 0
        else:
            scaler = joblib.load("Source/Models/scaler_file")
            b_comb = scaler.transform(b_comb)
            dl_pred = model_DL.predict(b_comb.reshape(b_comb.shape[0], 1, b_comb.shape[1]))
            dl_pred = dl_pred.reshape(dl_pred.shape[0], dl_pred.shape[2])
            score = np.mean(np.abs(dl_pred-b_comb), axis = 1)
            threshold = 0.29
            if score < threshold:
                op = 1
            else:
                op = -1
        Aop1 = Anomaly_output(op)
        return jsonify({"Output": Aop1})
            
api.add_resource(MakePrediction, '/predict_uni')    
api.add_resource(MakePrediction1, '/predict_multi')  
    
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000,debug=True)

