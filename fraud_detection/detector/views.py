from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import tensorflow as tf
import joblib
import numpy as np
import json

# Load the model and scaler
model = tf.keras.models.load_model('models/fraud_detection_model.h5')
scaler = joblib.load('models/scaler.pkl')


@csrf_exempt
def predict_fraud(request):
    if request.method == 'POST':
        try:
            # Parse the incoming JSON data
            data = json.loads(request.body)
            transaction = data['transaction']

            # Preprocess the transaction
            transaction = np.array(transaction).reshape(1, -1)
            transaction_scaled = scaler.transform(transaction)

            # Make prediction
            prediction_proba = model.predict(transaction_scaled)
            prediction = (prediction_proba > 0.5).astype(int).flatten()

            # Return the prediction
            return JsonResponse({"fraudulent": bool(prediction[0])})
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)
    else:
        return JsonResponse({"error": "Only POST requests are allowed"}, status=405)