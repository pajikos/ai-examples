# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

COPY requirements.txt /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Make port 7860 available to the world outside this container
EXPOSE 7860

# Run gradio_interface.py when the container launches
CMD ["python", "gradio_interface.py"]
