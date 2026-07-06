# 🎟️ ML-Powered Help Desk Ticket Classification System

A Python-based machine learning application that categorizes IT help desk tickets from their written descriptions, helping support teams route requests more efficiently.

##💡Business Problem
IT support teams oftenr recieve large volumes of tickets that meed to be manually reviewed and routed to the correct category or team. This project demonstrates how machine learning can help classify ticket descriptions automatically, reducing manual triage time and supporting faster issue resolution.

## Overview
This project processes helpdesk ticket text and predicts the appropriate category using a trained machine learning model. It simulates real-world IT support workflows by automating ticket classification.

## Features
- Machine learning model for ticket classification
- Data preprocessing and model training pipeline
- Real-time prediction capability
- Structured Python backend logic

##Project Struture

```text
ticket-classification-system-pink/
│
├── main.py                     # Application entry point
├── train_model.py              # Data preprocessing and model training
├── predict.py                  # Ticket prediction logic
├── helpdesk_tickets_dataset.csv
├── helpdesk_tickets_dataset_v2.csv
└── README.md

## Technologies Used
- Python
- pandas
- scikit-learn
- FastAPI

## How It Works
1. Ticket data is cleaned and prepared
2. A classification model is trained
3. New ticket input is processed
4. The system predicts the ticket category

## Future Improvements
- Deploy as a web API
- Improve model accuracy
- Add user interface for interaction
