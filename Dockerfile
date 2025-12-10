# Base stage
FROM python:3.10.6-slim

# Set the working directory
WORKDIR /prod

# Libraries required by OpenCV
RUN apt-get update
RUN apt-get install \
  'ffmpeg'\
  'libsm6'\
  'libxext6'  -y

# Leverage Docker's build cache
COPY requirements.txt requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --upgrade pip

# Copy application code
COPY lunar_crater_age_logic lunar_crater_age_logic/
COPY Fast_api Fast_api/
COPY setup.py setup.py

# Copy the model to container
COPY models models/

# Define the startup command
CMD uvicorn Fast_api.app:app --host 0.0.0.0 --port $PORT
