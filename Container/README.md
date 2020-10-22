# AI Techniques for Predictive Maintenance
## Anomaly Detection Machine Learning Model deployed using FLASK API and containerization using Docker
* Build and Run Docker in local computer __or__
* Pull the Docker image from the Docker hub by executing the command in the terminal __docker pull selvasundar/ai-pdm-docker:latest__
	* Run the Docker container by executing __docker run -p 127.0.0.1:8001:5000 selvasundar/ai-pdm-docker__
* Follow the steps in *Instructions_to_Execute* for Univariate, Multivariate Anomaly Detection and RUL Prediction

#### Note:
* Docker must be installed in the local computer
* In the command *docker run -p 127.0.0.1:8001:5000 selvasundar/ai-pdm-docker*, -p refers to Port Forwarding
* Use the same IP & Port (*127.0.0.1:8001*) for accessing the API