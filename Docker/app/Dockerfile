# Pull image from docker repository
FROM python:3.8-alpine
# Where I'll copy the image
COPY . /app
# The working directory will be where the image is
WORKDIR /app
# Commands to run
RUN pip install -r requirements.txt
CMD python app.py