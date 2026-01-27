from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.pipeline.prediction_pipeline import (
    MentalHealthData,
    MentalHealthPredictor
)

import uvicorn


# -----------------------------------
# App Initialization
# -----------------------------------

app = FastAPI(title="Mental Health Predictor")

# Static files (CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Predictor (loaded once)
predictor = MentalHealthPredictor()


# -----------------------------------
# Home Page
# -----------------------------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": None
        }
    )


# -----------------------------------
# Prediction Route
# -----------------------------------

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,

    Name: str = Form(...),
    Gender: str = Form(...),
    Age: int = Form(...),
    City: str = Form(...),
    Working_Professional_or_Student: str = Form(...),
    Profession: str = Form(None),

    Academic_Pressure: int = Form(None),
    Work_Pressure: int = Form(None),

    CGPA: float = Form(None),

    Study_Satisfaction: int = Form(None),
    Job_Satisfaction: int = Form(None),

    Sleep_Duration: str = Form(...),
    Dietary_Habits: str = Form(...),
    Degree: str = Form(...),

    Suicidal_Thoughts: str = Form(...),

    Work_Study_Hours: float = Form(...),

    Financial_Stress: int = Form(...),

    Family_History: str = Form(...)
):
    try:

        # -----------------------------------
        # Build Input Object
        # -----------------------------------

        user_data = MentalHealthData(

            Name=Name,
            Gender=Gender,
            Age=Age,
            City=City,

            Working_Professional_or_Student=Working_Professional_or_Student,

            Profession=Profession,

            Academic_Pressure=Academic_Pressure,
            Work_Pressure=Work_Pressure,

            CGPA=CGPA,

            Study_Satisfaction=Study_Satisfaction,
            Job_Satisfaction=Job_Satisfaction,

            Sleep_Duration=Sleep_Duration,
            Dietary_Habits=Dietary_Habits,
            Degree=Degree,

            Suicidal_Thoughts=Suicidal_Thoughts,

            Work_Study_Hours=Work_Study_Hours,

            Financial_Stress=Financial_Stress,

            Family_History=Family_History
        )


        # -----------------------------------
        # Convert to DataFrame
        # -----------------------------------

        df = user_data.get_input_dataframe()


        # -----------------------------------
        # Prediction
        # -----------------------------------

        prediction = predictor.predict(df)[0]


        # -----------------------------------
        # Label
        # -----------------------------------

        if prediction == 1:
            result = "⚠️ High Risk of Depression"
        else:
            result = "✅ Low Risk of Depression"


        # -----------------------------------
        # Render Page
        # -----------------------------------

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "result": result
            }
        )


    except Exception as e:

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "result": f"Error: {e}"
            }
        )


# -----------------------------------
# Run Server
# -----------------------------------

if __name__ == "__main__":

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
