# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn joblib numpy pydantic scikit-learn

# Expose the port that FastAPI will run on
EXPOSE 8000

# Define the command to run the FastAPI app with uvicorn
CMD ["uvicorn", "app2:app", "--host", "0.0.0.0", "--port", "8000"]

