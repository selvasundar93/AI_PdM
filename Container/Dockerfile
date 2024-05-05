FROM python:3.7
WORKDIR /deploy

COPY . . 
RUN pip install -r requirements.txt

EXPOSE 5000
CMD ["python", "app.py"]
