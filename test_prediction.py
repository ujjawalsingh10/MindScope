from src.pipeline.prediction_pipeline import (
    MentalHealthData,
    MentalHealthPredictor
)


def main():

    print("===== Testing Mental Health Prediction Pipeline =====")

    # Step 1: Create sample input (MATCH REAL DATA FORMAT)
    user_data = MentalHealthData(

        Name="Test User",
        Gender="Male",
        Age=23,
        City="Delhi",

        Working_Professional_or_Student="Student",
        Profession=None,

        Academic_Pressure=4.5,
        Work_Pressure=None,

        CGPA=8.2,

        Study_Satisfaction=3.5,
        Job_Satisfaction=None,

        Sleep_Duration="6-7 hours",
        Dietary_Habits="More Healthy",

        Degree="B.Tech",

        Suicidal_Thoughts="No",

        Work_Study_Hours=6.0,
        Financial_Stress=2.5,

        Family_History="No"
    )

    # Step 2: Convert to DataFrame
    input_df = user_data.get_input_dataframe()

    print("\nInput DataFrame:")
    print(input_df)

    # Step 3: Initialize predictor
    predictor = MentalHealthPredictor()

    # Step 4: Make prediction
    prediction = predictor.predict(input_df)

    print("\nPrediction Result:")
    print(prediction)

    print("\n===== Test Completed =====")


if __name__ == "__main__":
    main()
