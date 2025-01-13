# Use a lightweight Python image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy project files into the container
COPY . /app

# Copy the requirements file to the Docker image
COPY requirements.txt /app/requirements.txt

# Install dependencies from requirements.txt
RUN pip install -r requirements.txt

# Expose the port Flask runs on
EXPOSE 9696

# Command to run the Flask app
CMD ["python", "app.py"]


