# Use an official Python runtime as a parent image
FROM python:3.12

# Install any needed packages specified in requirements.txt
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

# Run app.py when the container launches
CMD ["python", "src/train.py"]
