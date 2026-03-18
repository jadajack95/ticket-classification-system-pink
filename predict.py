import joblib

#Load the model
model = joblib.load("models/ticket_category_model.pkl")

#Ask for ticket
ticket_text = input("Enter the helpdesk ticket text: ")

#Predict the category
prediction = model.predict([ticket_text])[0]

print(f"Predicted Category: {prediction}")
