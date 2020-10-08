# AI Techniques for Predictive Maintenance
## Instructions to Execute & get response from API
* Run app.py by executing the command __python app.py__ in terminal
* Use Browser, Jupyter Notebook or any Script to send & receive data from this API
### Univariate Anomaly Detection
* URL - Endpoint = "127.0.0.1:5000/Ano_Det_Uni"
* Load "{"rms":value,"mean":value}" as JSON data
	* Ex: *"{"rms": 0.072, "mean": 0.062}"*
* API will respond back with JSON data in the form {"Output": output_value}
	* Ex: *"{"Output": Normal}"*
### Multivariate Anomaly Detection
* URL - Endpoint = "127.0.0.1:5000/Ano_Det_Mutli"
* Load "{"Bearing1":value,"Bearing2":value,"Bearing3":value,"Bearing4":value}" as JSON data
	* Ex: *"{"Bearing1": 0.062, "Bearing2": 0.075, "Bearing3": 0.084, "Bearing4": 0.043}"*
* API will respond back with JSON data in the form {"Output": output_value}
	* Ex: *"{"Output": Normal}"*
* Three outputs are possible: Normal, Anomaly, No Proper Input
	* __Normal__ : Vibrations in Normal Range - Healthy State
	* __Anomaly__: Vibrations exceeds Normal Range - Unhealthy / Possibility of failure in near future
	* __No Proper Input__ : If all inputs are "0" - Sensor Fault / Machine is turned off