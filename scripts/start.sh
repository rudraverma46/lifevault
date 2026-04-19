#!/bin/bash
# Start LifeVault
echo "Starting LifeVault Web App..."
mkdir -p logs
source LIFEVAULT_1/bin/activate
python app.py > logs/app.log 2>&1 &
echo "App running in background. Logs in logs/app.log. Navigate to http://localhost:7860"
