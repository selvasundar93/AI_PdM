# AI Techniques for Predictive Maintenance
## Instructions to Execute & get response from API
* Run app.py by executing the command __python app.py__ in terminal
* Use Browser, Jupyter Notebook or any Script to send & receive data from this API
### Univariate Anomaly Detection
* URL - Endpoint = "127.0.0.1:5000/Ano_Det_Uni"
* Load "{"rms":value,"mean":value}" as JSON data
	* Ex: *"{"rms": 0.0782, "mean": 0.0619}"*
* API will respond back with JSON data in the form "{"Output": output_value}"
	* Ex: *"{"Output": Normal}"*
### Multivariate Anomaly Detection
* URL - Endpoint = "127.0.0.1:5000/Ano_Det_Multi"
* Load "{"Bearing1":value,"Bearing2":value,"Bearing3":value,"Bearing4":value}" as JSON data
	* Ex: *"{"Bearing1": 0.0619, "Bearing2": 0.0751, "Bearing3": 0.0825, "Bearing4": 0.0438}"*
* API will respond back with JSON data in the form "{"Output": output_value}"
	* Ex: *"{"Output": Normal}"*
### Remaining Useful Life Estimation
* URL - Endpoint = "127.0.0.1:5000/RUL_Predict"
* Load "{"Bearing1_RMS":value,"Bearing1_Kurt":value,"Bearing1_RMS_Prev":value,"Bearing1_Kurt_Prev":value}" as JSON data
	* Ex: *"{"Bearing1_RMS":0.0790,"Bearing1_Kurt":3.5062,"Bearing1_RMS_Prev":0.0789,"Bearing1_Kurt_Prev":3.5963}"*
* API will respond back with JSON data in the form "{"RUL_Class":output_value,"Fraction Failing":output_value, "RUL":output_value}"
	* Ex: *"{"RUL_Class":2,"Fraction Failing":"20-40%", "RUL":"60%"}"*

#### Notes
* Three outputs are possible (Anomaly Detection): Normal, Anomaly, No Proper Input
	* __Normal__ : Vibrations in Normal Range - Healthy State
	* __Anomaly__: Vibrations exceeds Normal Range - Unhealthy / Possibility of failure in near future
	* __No Proper Input__ : If all inputs are "0" - Sensor Fault / Machine is turned off
* For accurate prediction, input values must be given with atleast four decimal places (Ex: 0.0825)