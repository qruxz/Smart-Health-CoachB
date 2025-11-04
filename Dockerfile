# Use an official Python runtime as a base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /app

# Copy dependency files first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . .

# Expose the port (Railway uses PORT env var)
EXPOSE 5000

# Command to run the Flask app with Gunicorn for production
CMD gunicorn --bind 0.0.0.0:${PORT:-5000} app:app
