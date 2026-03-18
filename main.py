from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib

app = FastAPI()

# Load trained model
model = joblib.load("models/ticket_category_model.pkl")

# Mount static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": None,
            "confidence": None,
            "ticket_text": ""
        }
    )


@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request, ticket_text: str = Form(...)):
    prediction = model.predict([ticket_text])[0]

    probabilities = model.predict_proba([ticket_text])[0]
    confidence = max(probabilities) * 100

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "ticket_text": ticket_text
        }
    )