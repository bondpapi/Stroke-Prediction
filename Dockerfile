# Use a slim version of Python
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy all project files into the container's working directory
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port your Flask app runs on
EXPOSE 5000

# Run your app
CMD ["streamlit", "run", "app.py", "--server.port=5000", "--server.address=0.0.0.0"]

