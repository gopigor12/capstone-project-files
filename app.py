import pickle
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Step 1: Load the model and scaler when the app starts
with open("random_forest_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Step 2: Feature columns and weights (from your notebook)
feature_columns = [
    "Sender_Domain_Reputation", "URL_Legitimacy_Score", "Stress_Level",
    "Curiosity_Index", "Urgency_Trigger", "Prior_Training", "Email_Subject_Type_Fear",
    "Email_Subject_Type_Reward", "Attachment_Type_.zip", "Attachment_Type_none",
    "Email_Presentation_Well_Formed", "Email_Opened", "Attachment_Downloaded", "Reported_Phishing"
]

weights = {
    "Sender_Domain_Reputation": 0.3,
    "URL_Legitimacy_Score": 0.3,
    "Stress_Level": 0.3,
    "Curiosity_Index": 0.2,
    "Urgency_Trigger": 0.3,
    "Prior_Training": -0.05,
    "Email_Subject_Type_Fear": 0.25,
    "Email_Subject_Type_Reward": 0.25,
    "Attachment_Type_.zip": 0.25,
    "Attachment_Type_none": 0.1,
    "Email_Presentation_Well_Formed": -0.05
}

# Normalize weights to sum to 1
total_weight = sum(abs(w) for w in weights.values())
normalized_weights = {k: v / total_weight for k, v in weights.items()}

# Step 3: Define a route to process user inputs and predict
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            feature_columns = [
                "Sender_Domain_Reputation", "URL_Legitimacy_Score", "Stress_Level",
                "Curiosity_Index", "Urgency_Trigger", "Prior_Training", "Email_Subject_Type_Fear",
                "Email_Subject_Type_Reward", "Attachment_Type_.zip", "Attachment_Type_none",
                "Email_Presentation_Well_Formed", "Email_Opened", "Attachment_Downloaded", "Reported_Phishing"
                ]

            # Collect user input (0-10 scale, normalize to 0-1 for most features)
            user_data = {
                "Sender_Domain_Reputation": [float(input("Enter Sender Domain Reputation (0-10): ")) / 10],  # Scale to 0-1
                "URL_Legitimacy_Score": [float(input("Enter URL Legitimacy Score (0-10): ")) / 10],          # Scale to 0-1
                "Stress_Level": [float(input("Enter Stress Level (0-10): ")) / 10],                         # Scale to 0-1
                "Curiosity_Index": [float(input("Enter Curiosity Index (0-10): ")) / 10],                   # Scale to 0-1
                "Urgency_Trigger": [float(input("Enter Urgency Trigger (0 or 10): ")) / 10],                  # Scale to 0-1
                "Prior_Training": [float(input("Enter Prior Training (0 or 10): ")) / 10],
            }
            email_subject_type = input("Enter Email Subject Type (Fear/Reward/Curiosity): ").strip().lower()
            user_data["Email_Subject_Type_Fear"] = [1 if email_subject_type == "fear" else 0]
            user_data["Email_Subject_Type_Reward"] = [1 if email_subject_type == "reward" else 0]
            user_data["Email_Subject_Type_Curiosity"] = [1 if email_subject_type == "curiosity" else 0]
            attachment_type = input("Enter Attachment Type (.zip/none/.pdf): ").strip().lower()
            user_data["Attachment_Type_.zip"] = [1 if attachment_type == ".zip" else 0]
            user_data["Attachment_Type_none"] = [1 if attachment_type == "none" else 0]
            user_data["Attachment_Type_.pdf"] = [1 if attachment_type == ".pdf" else 0]

            user_data["Email_Presentation_Well_Formed"] = [1 if input("Is Email Well-Formed (yes/no): ").strip().lower() == "yes" else 0]

            # Convert to DataFrame
            user_input_df = pd.DataFrame(user_data)

            # Adjusted weights normalized to sum approximately to 1
            weights = {
                "Sender_Domain_Reputation": 0.3, # 1-10, 10 being a very bad reputation
                "URL_Legitimacy_Score": 0.3, # 1-10, 10 being a very bad reputation
                "Stress_Level": 0.3, # 1-10, 10 being a very high stress level
                "Curiosity_Index": 0.2, # 1-10, 10 being a very curious
                "Urgency_Trigger": 0.3, # 1-10, 10 being drawn to urgency
                "Prior_Training": -0.05, # 1-10, 10 being most training
                "Email_Subject_Type_Fear": 0.25, # 1-10, 10 being falling trap to fear more
                "Email_Subject_Type_Reward": 0.25, # 1-10, 10 being falling trap to fear more
                "Email_Subject_Type_Curiosity": 0.15, # 1-10, 10 being more curious
                "Attachment_Type_.zip": 0.25,
                "Attachment_Type_.pdf": 0.2,
                "Attachment_Type_none": 0.1,
                "Email_Presentation_Well_Formed": -0.05,
                "Email_Presentation_Poorly_Formed": 0.25
            }

            # Normalize weights to sum to 1
            total_weight = sum(abs(w) for w in weights.values())
            normalized_weights = {k: v / total_weight for k, v in weights.items()}

            # Debug: Print normalized weights
            print("Normalized Weights:", normalized_weights)

            # Ensure all expected columns are present
            for col in weights.keys():
                if col not in user_input_df.columns:
                    user_input_df[col] = 0  # Fill missing columns with 0

            # Debug: Print user input data after ensuring all features are present
            print("User Input DataFrame:")
            print(user_input_df)

            # Calculate phishing risk score
            risk_score = 0
            for feature, weight in normalized_weights.items():
                risk_score += user_input_df[feature][0] * weight
                # Debug: Log contribution of each feature to risk score
                print(f"Feature: {feature}, Value: {user_input_df[feature][0]}, Weight: {weight}, Contribution: {user_input_df[feature][0] * weight}")

            # Convert risk score to percentage (between 0 and 100)
            risk_percentage = max(0, min(risk_score * 100, 100))

            # Step 3.3: Use the model to predict if needed (optional, for example classification)
            # We can integrate model prediction if needed, but for now we use the calculated risk score
            prediction = model.predict(user_input_df)
            prediction_proba = model.predict(user_input_df)[0]

            return render_template("frontend.html", risk_percentage=risk_percentage, prediction=prediction[0], prediction_proba=prediction_proba)

        except Exception as e:
            # Step 3.4: Handle errors
            return render_template("frontend.html", risk_percentage=None, error=f"An error occurred: {str(e)}")

    # Render the form for GET requests
    return render_template("frontend.html", risk_percentage=None)

if __name__ == "__main__":
    app.run(debug=True)
