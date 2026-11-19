import csv
import pandas as pd
import numpy as np
import streamlit as st
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load training and testing data
@st.cache_data
def load_data():
    training = pd.read_csv('Training.csv')
    testing = pd.read_csv('Testing.csv')
    training.columns = training.columns.str.strip()
    testing.columns = testing.columns.str.strip()
    training['prognosis'] = training['prognosis'].str.strip()
    testing['prognosis'] = testing['prognosis'].str.strip()
    return training, testing

training, testing = load_data()
cols = training.columns[:-1]
x = training[cols]
y = training['prognosis']

# Mapping strings to numbers
le = preprocessing.LabelEncoder()
le.fit(y)
y_encoded = le.transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.33, random_state=42)

# Train a robust model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(x_train, y_train)

# Global dictionaries for lookup
severityDictionary = {}
description_list = {}
precautionDictionary = {}

def getDescription():
    global description_list
    try:
        with open('Indicator_Description.csv', encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                if len(row) >= 2:
                    description_list[row[0].strip()] = row[1].strip()
    except Exception as e:
        print("Error loading descriptions:", e)

def getSeverityDict():
    global severityDictionary
    try:
        with open('Indicator_severity.csv', encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                if len(row) >= 2:
                    try:
                        severityDictionary[row[0].strip()] = int(row[1].strip())
                    except ValueError:
                        pass
    except Exception as e:
        print("Error loading severity:", e)

def getprecautionDict():
    global precautionDictionary
    try:
        with open('Indicator_precaution.csv', encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                if len(row) >= 5:
                    precautionDictionary[row[0].strip()] = [row[1].strip(), row[2].strip(), row[3].strip(), row[4].strip()]
    except Exception as e:
        print("Error loading precautions:", e)

# Function to predict disease based on selected symptoms
def predict_disease(selected_symptoms):
    input_vector = pd.Series(0, index=cols)
    for symptom in selected_symptoms:
        if symptom in input_vector:
            input_vector[symptom] = 1
    prediction_encoded = clf.predict([input_vector])[0]
    predicted_disease = le.inverse_transform([prediction_encoded])[0]
    return predicted_disease

# Main Streamlit app
def main():
    st.title('Pain Management & Symptom Prediction App')

    # Load data lookups
    getDescription()
    getSeverityDict()
    getprecautionDict()

    st.header('User Input')

    # Name input
    name = st.text_input('Your Name', value='Rahul Verma')

    # How many days experiencing input
    num_days = st.number_input('How many days you are experiencing symptoms', min_value=1, value=1)

    # Symptom selection
    st.header('Select Your Symptoms')
    selected_symptoms = st.multiselect('Choose all symptoms that apply:', list(cols))

    # Predict disease button
    if st.button('Predict'):
        if selected_symptoms:
            predicted_disease = predict_disease(selected_symptoms)
            st.success(f'Hello, {name}!')
            st.header('Predicted Disease')
            st.markdown(f'**Based on your symptoms, you may have:** `{predicted_disease}`')
            
            st.header('Description')
            description = description_list.get(predicted_disease, 'No description available for this condition.')
            st.write(description)

            precaution_list = precautionDictionary.get(predicted_disease, [])
            st.header('Precaution Measures')
            if precaution_list:
                st.write('Take the following measures:')
                for i, measure in enumerate(precaution_list):
                    if measure:
                        st.write(f"{i+1}. {measure}")
            else:
                st.write('No specific precautions listed.')
        else:
            st.warning('Please select at least one symptom before predicting.')

if __name__ == '__main__':
    main()