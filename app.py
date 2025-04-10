from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from source.pipeline.prediction_pipeline import CustomData, PredictPipeline

# Initialize Flask app
application = Flask(__name__)
app = application

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for prediction
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        # Gather data from the form
        data = CustomData(
            year=float(request.form.get('year') or 2000),
            miles=float(request.form.get('miles') or 0),
            brand=request.form.get('brand'),
            color_exterior=request.form.get('color_exterior'),
            number_of_owners=float(request.form.get('number_of_owners') or 0),
            color_interior=request.form.get('color_interior'),
            accidents=float(request.form.get('accidents') or 0)
        )

        # Get prediction data
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        # Prediction pipeline
        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print("After Prediction")

        # Return the result on the same page
        return render_template('home.html', results=results[0])

    # For GET request, render home page
    return render_template('home.html')

# Running the application with Waitress for production (locally can still use Flask’s built-in server)
if __name__ == "__main__":
    from waitress import serve
    print("Starting application...")
    serve(application, host="0.0.0.0", port=8080)

