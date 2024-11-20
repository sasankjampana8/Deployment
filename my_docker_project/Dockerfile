# Start with a base image
FROM ubuntu:latest

# Add metadata to the image
# LABEL maintainer="your-email@example.com"

# Install dependencies
RUN apt-get update && apt-get install -y python3

# Copy the application code
COPY hello.py /app/hello.py

# Set the working directory
WORKDIR /app

# Run the application
CMD ["python3", "hello.py"]
