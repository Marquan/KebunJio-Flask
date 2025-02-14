# Use official Python image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Expose the port Flask runs on
EXPOSE 8080

# Start Flask when the container runs
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]