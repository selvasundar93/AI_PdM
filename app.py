import pickle
import numpy as np
from flask import Flask, jsonify, request
from flask_restful import Api, Resource

app = Flask(__name__)
api = Api(app)

model_rms = pickle.load(open('Source/Models/lof_rms_trained_model.pkl', 'rb'))
model_mean = pickle.load(open('Source/Models/lof_mean_trained_model.pkl', 'rb'))

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
        if op==1:
            Aop = "Normal"
        elif op==-1:
            Aop = "Anomaly"
        else:
            Aop = "No Proper Input"
        return jsonify({"Output": Aop})
api.add_resource(MakePrediction, '/predict')    
  
    
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000,debug=True)
