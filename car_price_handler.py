import os
import joblib
import numpy as np
import pandas as pd
from ts.torch_handler.base_handler import BaseHandler

class CarPriceHandler(BaseHandler):
    """
    Custom TorchServe handler for car price prediction.
    """

    def _init_(self):
        super(CarPriceHandler, self)._init_()
        self.initialized = False

    def initialize(self, context):
        """
        Load the model, scaler, and one-hot encoder during initialization.
        """
        # Get model directory
        properties = context.system_properties
        model_dir = properties.get("model_dir")

        # Load the pre-trained XGBoost model, scaler, and one-hot encoder
        self.model = joblib.load(os.path.join(model_dir, "XGBoost.joblib"))
        self.scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
        self.ohe = joblib.load(os.path.join(model_dir, "ohe.joblib"))
        
        self.initialized = True

    def preprocess(self, data):
        """
        Preprocess the input JSON into the format required by the model.
        """
        # Extract input from JSON
        row = data[0]["body"]["row"]
        cols = data[0]["body"]["cols"]

        # Convert input row into a DataFrame
        df = pd.DataFrame([row], columns=cols)
        
        # Get numeric and categorical columns
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        car_num_cols = list(df.select_dtypes(include=numerics).columns)
        car_cat_cols = list(df.select_dtypes(exclude=numerics).columns)

        # Scale numeric features
        df[car_num_cols] = self.scaler.transform(df[car_num_cols])

        # One-hot encode categorical features
        car_ohe = self.ohe.transform(df[car_cat_cols])

        # Get feature names and create a DataFrame for the encoded features
        feature_names = self.ohe.get_feature_names_out(input_features=car_cat_cols)
        car_df_ohe = pd.DataFrame(car_ohe, columns=feature_names)

        # Combine scaled numeric and encoded categorical features
        df = df.drop(car_cat_cols, axis=1)
        df = pd.concat([df, car_df_ohe], axis=1)

        return df

    def inference(self, data):
        """
        Perform inference using the pre-trained XGBoost model.
        """
        # Convert input DataFrame to numpy array for XGBoost
        data_np = data.to_numpy()
        
        # Make prediction
        prediction = self.model.predict(data_np)
        
        return prediction

    def postprocess(self, inference_output):
        """
        Convert the raw model output into a user-friendly format.
        """
        # Return the predicted car price
        return {"predicted_price": float(inference_output[0])}