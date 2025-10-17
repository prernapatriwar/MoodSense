#!/bin/bash

# Create main project directory
#mkdir -p mood_microcoach

# Create subdirectories for Flask backend, model, frontend, and utility scripts
mkdir -p app
mkdir -p model
mkdir -p static
mkdir -p templates
mkdir -p scripts
mkdir -p data

# Create sample files inside each directory

# Flask app
touch app/app.py
touch app/requirements.txt

# Model folder
touch model/sentiment_model.pkl
touch model/tfidf.pkl
touch model/label_encoder.pkl

# Frontend folder
touch templates/index.html
touch static/style.css

# Utility scripts
touch scripts/train_baseline.py
touch scripts/predict.py

# Data folder (you can add your dataset here later)
touch data/emotions_train.csv

# Print success message
echo "Folder structure created successfully!"