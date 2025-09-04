# Pull image from docker repository
FROM python:3.8-alpine
# The working directory will be where the image is
WORKDIR /code
# Create some environment variables
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

COPY . .
# Commands to run
RUN pip install -r requirements.txt
# port number expose from the container
EXPOSE 5000
CMD ["flask","run"]