#!/bin/bash

# Check if Python is installed
if ! command -v python3.11 &> /dev/null
then
    echo "Python3 is not installed. Please install it first."
    exit 1
fi

# Define the virtual environment name
ENV_NAME="venv"

# Create the virtual environment
if [ ! -d "$ENV_NAME" ]; then
    echo "Creating virtual environment: $ENV_NAME"
    python3 -m venv $ENV_NAME
else
    echo "Virtual environment '$ENV_NAME' already exists."
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source $ENV_NAME/bin/activate

# Confirm activation
if [ "$VIRTUAL_ENV" != "" ]; then
    echo "Virtual environment '$ENV_NAME' activated successfully."
else
    echo "Failed to activate virtual environment."
fi

# Load environment variables from .env file
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo ".env file not found!"
fi

pip install -r requirements.txt
python main.py