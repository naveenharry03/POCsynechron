from flask import Flask, request, jsonify , Response
import tensorflow as tf
import pandas as pd
import numpy as np
import joblib
import json

# Create the app object
app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('POC36420')

# Load the scaler
scaler = joblib.load('scaler.pkl')

# Define a post method for our API.
@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON object containing the data sent by the client
    data = request.get_json()

    # Convert the list of dictionaries to a pandas DataFrame
    data_df = pd.DataFrame(data)

    # Convert the DataFrame to a numpy array
    data_np = data_df.to_numpy()

    # Convert the data to scale it using the trained scaler
    data_np = scaler.transform(data_np)

    # # Convert the data to float32
    data_np = data_np.astype(np.float32)

    # Reshape the data to match the input shape that the model expects
    data_np = data_np.reshape(1, data_np.shape[0], data_np.shape[1])

    # Make a prediction using the model
    prediction = model(data_np)

    # Inverse transform the prediction
    prediction = scaler.inverse_transform(prediction)    

    # Get the first prediction from the array of predictions
    output = prediction[0]

    # Convert the EagerTensor to a numpy array and then to a list
    output_list = output.tolist()

    # Define the feature names
    feature_names = test_copy_first_5.columns.tolist()

    # Pair each feature name with its corresponding value
    output_dict = dict(zip(feature_names, output_list))

    # Convert the dictionary to a JSON string without sorting the keys
    output_json = json.dumps(output_dict, sort_keys=False)

    # Return the JSON string as a response
    return Response(output_json, mimetype='application/json')

    # Run the app
if __name__ == '__main__':
    #app.run()
    app.run(host='0.0.0.0', port=5000, debug=True)
