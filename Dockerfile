# Use an official Python image as the base
FROM python:3.11

# Set the working directory inside the container
WORKDIR /app

# Copy the application files to the container
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Run app_store.py first to process repositories
RUN python app_store.py

# Command to run the Streamlit app
CMD ["streamlit", "run", "app_main.py", "--server.port=8501", "--server.address=0.0.0.0"]

