# Use an official, lightweight Python image as the base
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker's caching
COPY requirements.txt requirements.txt

# Install all the necessary Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and necessary assets into the container
COPY ./app /app/app/
COPY ./model_assets /app/model_assets/
COPY ./geo_mapping.json /app/geo_mapping.json
COPY ./active_to_expired_mapping.json /app/active_to_expired_mapping.json

# Tell Docker that your application listens on port 8080
EXPOSE 8080

# The command to run your FastAPI application when the container starts
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]