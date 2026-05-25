# Install a light weight pythoh image
FROM python:3.10-slim

# Set the working directory inside the conatiner to /app
WORKDIR /app

# Copy requirements.txt inside the container
COPY requirements.txt .

# Install the dependencies inside the container
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code to the container
COPY . .

# Expose port
EXPOSE 8501

# Start command
CMD [ "streamlit", "rum", "app.py", "--server.port=8501", "--server.address=0.0.0.0" ]
