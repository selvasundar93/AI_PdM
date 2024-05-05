# AI Techniques for Predictive Maintenance
## Instructions to Execute 
* Run app.py by executing the command __python app.py__ in terminal
* Use Jupyter Notebook or any Script to send & receive data from this API
### Univariate Anomaly Detection
* URL - Endpoint = "127.0.0.1:5000/Ano_Det_Uni"
* Load __"{"rms":value,"mean":value}"__ as JSON data
	* Ex: *"{"rms": 0.0782, "mean": 0.0619}"*
* API will respond back with JSON data in the form "{"Output": output_value}"
	* Ex: *"{"Output": Normal}"*
### Multivariate Anomaly Detection
* URL - Endpoint = "127.0.0.1:5000/Ano_Det_Multi"
* Load __"{"Bearing1":value,"Bearing2":value,"Bearing3":value,"Bearing4":value}"__ as JSON data
	* Ex: *"{"Bearing1": 0.0619, "Bearing2": 0.0751, "Bearing3": 0.0825, "Bearing4": 0.0438}"*
* API will respond back with JSON data in the form "{"Output": output_value}"
	* Ex: *"{"Output": Normal}"*
### Remaining Useful Life Estimation
* URL - Endpoint = "127.0.0.1:5000/RUL_Predict"
* Load __"{"Bearing1_RMS":value,"Bearing1_Kurt":value,"Bearing1_RMS_Prev":value,"Bearing1_Kurt_Prev":value}"__ as JSON data
	* Ex: *"{"Bearing1_RMS":0.0790,"Bearing1_Kurt":3.5062,"Bearing1_RMS_Prev":0.0789,"Bearing1_Kurt_Prev":3.5963}"*
* API will respond back with JSON data in the form "{"RUL_Class":output_value,"Fraction Failing":output_value, "RUL":output_value}"
	* Ex: *"{"RUL_Class":2,"Fraction Failing":"20-40%", "RUL":"60%"}"*

### Input - Description
Input|Detail
-----|------
rms|__Root Mean Square__ Value of 1 sec Vibration Signal of __Bearing 1__
mean|__Mean__ Value of 1 sec Vibration Signal of __Bearing 1__
Bearing1|__Mean__ Value of 1 sec Vibration Signal of __Bearing 1__
Bearing2|__Mean__ Value of 1 sec Vibration Signal of __Bearing 2__
Bearing3|__Mean__ Value of 1 sec Vibration Signal of __Bearing 3__
Bearing4|__Mean__ Value of 1 sec Vibration Signal of __Bearing 4__
Bearing1_RMS|__Root Mean Square__ Value of 1 sec Vibration Signal of __Bearing 1__ at *time t*
Bearing1_Kurt|__Kurtosis__ Value of 1 sec Vibration Signal of __Bearing 1__ at *time t*
Bearing1_RMS_Prev|__Root Mean Square__ Value of 1 sec Vibration Signal of __Bearing 1__ at *time t-1*
Bearing1_Kurt_Prev|__Kurtosis__ Value of 1 sec Vibration Signal of __Bearing 1__ at *time t-1*

#### Note:
* Three outputs are possible (Anomaly Detection): Normal, Anomaly, No Proper Input
	* __Normal__ : Vibrations in Normal Range - Healthy State
	* __Anomaly__: Vibrations exceeds Normal Range - Unhealthy / Possibility of failure in near future
	* __No Proper Input__ : If all inputs are "0" - Sensor Fault / Machine is turned off
* For accurate prediction, input values must be given with atleast four decimal places (Ex: 0.0825)
* Refer Tests folder for examples
* Refer Tests/FE_Data folder for feature engineered dataset 
	* For Univariate and Multivariate Anomaly Detection, refer Anomaly_Detection_Dataset file
	* For Remaining Useful Life Prediction, refer RUL_Dataset file
	