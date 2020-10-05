# AI Techniques for Predictive Maintenance
## Instructions to Execute & get response from API
* Run app.py by executing the command __python app.py__ in terminal
* URL = "127.0.0.1:5000/predict"
* Load "{"rms":value,"mean":value}" as JSON data
** Ex: *"{"rms": 0.072, "mean":0.062}"*
* Use Jupyter Notebook or Any Script to send & receive data from this API
* API will respond back with JSON data in the form {"Output": output_value}
** Ex: *"{"Output": Normal}"*
* Three outputs are possible Normal, Anomaly, No Proper Input