# Use a minimal base image with Python 3.9 installed
FROM python:3.9

# Set the working directory inside the container to /app
WORKDIR /app

# Copy all files from the current directory on the host to the container's /app directory
COPY . .

# Install Python dependencies listed in requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

RUN useradd -m -u 1000 user
USER user

# Define the command to run the Streamlit app on port "8501" and make it accessible externally
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
