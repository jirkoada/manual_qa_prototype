# Use an official Python runtime as a base image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install the required Python packages
RUN pip install --no-cache-dir langchain openai tiktoken faiss-cpu unstructured flask
ENV OPENAI_API_KEY '' # ADD API KEY BEFORE EXECUTION
RUN python create_vector_store.py

# Expose the port your Flask app is listening on
EXPOSE 5000

# Define the command to run your Flask app using the Flask development server
CMD python app.py
