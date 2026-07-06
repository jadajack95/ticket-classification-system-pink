# 🎟️ ML-Powered Help Desk Ticket Classification System

A Python-based machine learning application that categorizes IT help desk tickets from their written descriptions, helping support teams route requests more efficiently.

## 💡 Business Problem

IT support teams often receive large volumes of tickets that need to be manually reviewed and routed to the correct category or team. This project demonstrates how machine learning can help classify ticket descriptions automatically, reducing manual triage time and supporting faster issue resolution.

## 🔎 Overview

This project processes help desk ticket text and predicts the appropriate category using a trained machine learning model. It simulates real-world IT support workflows by automating ticket classification.

## ✨ Features

- Machine learning model for ticket classification
- Data preprocessing and model training pipeline
- Real-time prediction capability
- Structured Python backend logic

## 📁 Project Structure

```text
ticket-classification-system-pink/
│
├── main.py                     # Application entry point
├── train_model.py              # Data preprocessing and model training
├── predict.py                  # Ticket prediction logic
├── helpdesk_tickets_dataset.csv
├── helpdesk_tickets_dataset_v2.csv
└── README.md
