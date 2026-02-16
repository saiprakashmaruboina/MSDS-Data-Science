import pandas as pd
from pycaret.classification import load_model, predict_model

class ChurnPredictor:

    def __init__(self, model_name="churn_model"):
        # Load saved PyCaret model
        self.model = load_model(model_name)

    def predict_churn_probability(self, df):
        # Generate predictions with probabilities
        predictions = predict_model(self.model, data=df)

        # Return probability column (PyCaret 3 uses prediction_score)
        return predictions["prediction_score"]


if __name__ == "__main__":

    # Load new data
    new_data = pd.read_csv("new_churn_data.csv")

    # Initialize predictor
    predictor = ChurnPredictor()

    # Get probabilities
    probabilities = predictor.predict_churn_probability(new_data)

    print("Churn Probabilities:")
    print(probabilities)
