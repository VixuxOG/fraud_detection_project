import sys
print(f"--- Running Python from: {sys.executable} ---")

import os
import pandas as pd
# ... rest of your code
import os
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import logging

import sys
import pprint # Import pprint to make the path list easier to read

print(f"--- Running Python from: {sys.executable} ---")
print("--- Python Search Path (sys.path): ---")
pprint.pprint(sys.path) # Print the list of paths
print("--- End of sys.path ---")

# Now try to import pandas AFTER printing the path
import os
try:
    import pandas as pd
    print("--- Successfully imported pandas ---")
except ModuleNotFoundError as e:
    print(f"--- Failed to import pandas: {e} ---")
    # You might want to exit here if pandas is essential right away
    # sys.exit(1) # Optional: Stop script if pandas fails

import numpy as np # Keep other imports
import joblib
# ... rest of your code

# --- Configuration ---
MODEL_DIR = 'models'
PREPROCESSOR_DIR = 'preprocessors'
MODEL_NAME = 'best_model.keras'
PREPROCESSOR_NAME = 'preprocessor.joblib'
ALLOWED_EXTENSIONS = {'csv'}

# Define the features exactly as used during training
# Make sure these lists are IDENTICAL to your training script's lists
NUMERICAL_FEATURES = [
    'amount', 'transaction_amount_vs_sender_history', 'geographic_disparity',
    'transaction_time_of_day', 'receiver_account_age', 'receiver_transaction_history',
    'session_duration', 'authentication_attempts', 'input_timing_consistency',
    'app_switching_frequency', 'keyboard_input_speed', 'input_pause_patterns',
    'screen_active_time', 'background_data_usage', 'authentication_attempt_count',
    'pin_entry_speed', 'transaction_velocity', 'failed_transaction_count',
    'handle_similarity_score', 'hour', 'day_of_week'
]

CATEGORICAL_FEATURES = [
    'merchant_category_code', 'session_source', 'authorization_method',
    'transaction_type', 'handle_verification_status'
]

# Combine features (ensure order matches preprocessor expectations if it matters)
# The order matters for the DataFrame slicing before preprocessor.transform()
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# --- Initialization ---
app = Flask(__name__)
app.secret_key = os.urandom(24) # Needed for flashing messages if you add them

# Configure logging
logging.basicConfig(level=logging.INFO)

# --- Load Model and Preprocessor ---
try:
    model_path = os.path.join(MODEL_DIR, MODEL_NAME)
    preprocessor_path = os.path.join(PREPROCESSOR_DIR, PREPROCESSOR_NAME)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"Preprocessor file not found at {preprocessor_path}")

    model = tf.keras.models.load_model(model_path)
    preprocessor = joblib.load(preprocessor_path)
    logging.info("Model and preprocessor loaded successfully.")

except FileNotFoundError as e:
    logging.error(f"Error loading files: {e}")
    # Handle the error appropriately - maybe exit or disable prediction
    model = None
    preprocessor = None
except Exception as e:
    logging.error(f"An unexpected error occurred during loading: {e}")
    model = None
    preprocessor = None

# --- Helper Functions ---
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Flask Routes ---
@app.route('/')
def index():
    """Renders the main upload page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles CSV upload, preprocessing, prediction, and returns results."""
    if model is None or preprocessor is None:
        return jsonify({'success': False, 'error': 'Model or preprocessor not loaded. Check server logs.'}), 500

    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        try:
            # Read CSV directly from file stream
            df = pd.read_csv(file)
            logging.info(f"Successfully read CSV file: {secure_filename(file.filename)}")

            # --- Data Validation ---
            required_columns = ALL_FEATURES + ['timestamp']
            # Remove engineered features from initial check
            required_input_columns = [col for col in required_columns if col not in ['hour', 'day_of_week']]

            missing_cols = [col for col in required_input_columns if col not in df.columns]
            if missing_cols:
                 logging.warning(f"Missing required columns: {missing_cols}")
                 return jsonify({'success': False, 'error': f'Missing required columns: {", ".join(missing_cols)}'}), 400

            original_df = df.copy() # Keep original data for displaying results

            # --- Feature Engineering ---
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                if df['timestamp'].isnull().any():
                    raise ValueError("Invalid datetime format found in 'timestamp' column.")
                df['hour'] = df['timestamp'].dt.hour
                df['day_of_week'] = df['timestamp'].dt.dayofweek
                # Ensure 'transaction_time_of_day' is present if it wasn't engineered here
                # Assuming 'transaction_time_of_day' was already in the input CSV based on list
            except Exception as e:
                 logging.error(f"Error during feature engineering: {e}")
                 return jsonify({'success': False, 'error': f"Error processing timestamp column: {e}"}), 400

            # --- Handle Missing Values (Before Preprocessing) ---
            # Strategy: Fill numerical with 0, categorical with 'Missing'
            # Note: Ensure 'Missing' is handled by your OneHotEncoder (handle_unknown='ignore')
            df[NUMERICAL_FEATURES] = df[NUMERICAL_FEATURES].fillna(0)
            df[CATEGORICAL_FEATURES] = df[CATEGORICAL_FEATURES].fillna('Missing')
            logging.info("Handled missing values.")

            # --- Select Features in Correct Order ---
            try:
                X_predict = df[ALL_FEATURES]
            except KeyError as e:
                 logging.error(f"Feature selection error - likely engineered feature missing: {e}")
                 return jsonify({'success': False, 'error': f"Internal error selecting features after engineering: {e}"}), 500


            # --- Preprocessing ---
            X_processed = preprocessor.transform(X_predict)
            # Check if sparse matrix and convert if necessary (depends on your preprocessor)
            if hasattr(X_processed, "toarray"):
                 X_processed = X_processed.toarray()
            logging.info("Data preprocessing complete.")

            # --- Reshape for 1D CNN ---
            # Shape should be (num_samples, num_features, 1)
            X_reshaped = X_processed.reshape(X_processed.shape[0], X_processed.shape[1], 1)
            logging.info(f"Data reshaped to: {X_reshaped.shape}")

            # --- Prediction ---
            y_pred_prob = model.predict(X_reshaped)
            # Apply threshold (e.g., 0.5)
            y_pred = (y_pred_prob > 0.5).astype(int).flatten() # Flatten to 1D array
            logging.info("Prediction complete.")

            # --- Extract Results ---
            original_df['is_fraud_prediction'] = y_pred
            fraudulent_df = original_df[original_df['is_fraud_prediction'] == 1].copy()

            fraud_count = len(fraudulent_df)

            # Select columns to display in the frontend table
            display_columns = [
                'transaction_id', 'user_id', 'merchant_id', 'amount',
                'timestamp', 'description'
                # Add any other columns from original_df you want to show
            ]
            # Ensure all display columns exist, handle missing ones gracefully
            actual_display_columns = [col for col in display_columns if col in fraudulent_df.columns]
            fraud_details_df = fraudulent_df[actual_display_columns]

            # Convert datetime to string for JSON serialization
            if 'timestamp' in fraud_details_df.columns:
                fraud_details_df['timestamp'] = fraud_details_df['timestamp'].astype(str)

            # Convert NaN/NaT to None (which becomes null in JSON) for better handling
            fraud_details_df = fraud_details_df.replace({np.nan: None})

            fraud_details = fraud_details_df.to_dict(orient='records')

            logging.info(f"Found {fraud_count} fraudulent transactions.")

            return jsonify({
                'success': True,
                'fraud_count': fraud_count,
                'fraudulent_transactions': fraud_details
            })

        except pd.errors.ParserError:
             logging.warning("Invalid CSV file format.")
             return jsonify({'success': False, 'error': 'Invalid CSV file format.'}), 400
        except FileNotFoundError as e: # Should be caught at startup, but just in case
             logging.error(f"Model/Preprocessor file not found during prediction: {e}")
             return jsonify({'success': False, 'error': f'Internal Server Error: {e}'}), 500
        except ValueError as e: # Catch specific errors like datetime parsing
             logging.error(f"Data processing error: {e}")
             return jsonify({'success': False, 'error': f"Data processing error: {e}"}), 400
        except Exception as e:
             logging.error(f"An unexpected error occurred during prediction: {e}", exc_info=True)
             return jsonify({'success': False, 'error': f'An internal error occurred: {e}'}), 500

    else:
        return jsonify({'success': False, 'error': 'Invalid file type. Please upload a CSV file.'}), 400

# --- Run Application ---
if __name__ == '__main__':
    # Set host='0.0.0.0' to make it accessible on your network, otherwise default is 127.0.0.1
    app.run(debug=True, host='127.0.0.1', port=5000)