import pandas as pd
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

app = Flask(__name__)

# Load data
data = pd.read_excel(r"C:\Users\allan\Desktop\Legal.xlsx")

# Fit the LabelEncoder for categorical columns
label_encoders = {}
for col in ['appeal_district', 'trial_district', 'offence']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Prepare the feature matrix and target vector
X = data[['appeal_district', 'trial_district', 'offence', 'no_female_appealant', 'no_public_witness']]
y = (data['scn_decision'] == 'Granted').astype(int)  # 1 if granted (win), 0 otherwise

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the trained model and label encoders
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

@app.route('/')
def index():
    appeal_districts = label_encoders['appeal_district'].classes_
    trial_districts = label_encoders['trial_district'].classes_
    offenses = label_encoders['offence'].classes_
    return render_template('index.html', appeal_districts=appeal_districts, trial_districts=trial_districts, offenses=offenses)

@app.route('/predict', methods=['POST'])
def predict():
    appeal_district = request.form['appeal_district']
    trial_district = request.form['trial_district']
    offence = request.form['offense']
    
    # Validate and convert inputs for no_female_appealant and no_public_witness
    try:
        no_female_appealant = int(request.form['no_female_appealant']) if request.form['no_female_appealant'] else 0
        no_public_witness = int(request.form['no_public_witness']) if request.form['no_public_witness'] else 0
    except ValueError:
        return render_template('result.html', result="Error", message="Please enter valid numbers for the number of female appellants and public witnesses.")

    # Load the trained model and label encoders
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)

    # Create input data
    input_data = pd.DataFrame({
        'appeal_district': [appeal_district],
        'trial_district': [trial_district],
        'offence': [offence],
        'no_female_appealant': [no_female_appealant],
        'no_public_witness': [no_public_witness]
    })

    # Encode the categorical features with error handling
    try:
        input_data['appeal_district'] = label_encoders['appeal_district'].transform(input_data['appeal_district'])
        input_data['trial_district'] = label_encoders['trial_district'].transform(input_data['trial_district'])
        input_data['offence'] = label_encoders['offence'].transform(input_data['offence'])
    except ValueError as e:
        return render_template('result.html', result="Error", message=str(e))

    # Predict probability of winning
    probability_of_winning = model.predict_proba(input_data)[0][1]

    # Adjust probability if the appeal and trial districts match
    if appeal_district == trial_district:
        probability_of_winning += 0.10  # Boost the probability by 10% if districts match
        probability_of_winning = min(probability_of_winning, 1.0)  # Ensure it doesn't exceed 100%

    predicted_outcome = model.predict(input_data)[0]

    # Determine the message based on probability
    if probability_of_winning >= 0.80:
        message = "Your chances are high. You can go home!"
    else:
        # Retrieve the sentence associated with the offence from the excel sheet
        sentence = data.loc[data['offence'] == label_encoders['offence'].transform([offence])[0], 'sentence'].values[0]
        message = f"You have a lower chance of winning. {sentence}"

    # Send data to the result template
    return render_template('result.html',
                           result="Win" if predicted_outcome == 1 else "Loss",
                           probability=probability_of_winning,
                           message=message)

if __name__ == '__main__':
    app.run(debug=True)
