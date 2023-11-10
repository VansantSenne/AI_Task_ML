import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import seaborn as sns

# Laden van de dataset
dataFile = 'auto-mpg.data'
df = pd.read_csv(dataFile, sep=r'\s+', skiprows=1, names=['displacement', 'mpg', 'cylinders', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name'])

# Vervangen van '?' met NaN en invullen van ontbrekende waarden
df.replace('?', np.nan, inplace=True)
df = df.fillna(df.mode().iloc[0])

# One-hot encoding voor de 'car_name' kolom
df = pd.get_dummies(df, columns=['car_name'])

laatste_kolom = df.columns[1]

X = df.drop(columns=[laatste_kolom])
y = df[laatste_kolom]

# Split van de dataset in trainings- en testsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Oversampling van de trainingsdata met RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

# Streamlit-applicatie
st.title("Model Evaluatie Applicatie")

# Opties om tussen modellen te schakelen
selected_model = st.sidebar.selectbox("Selecteer Model", ["Gradient Boosting", "Support Vector Machine (SVM)", "Random Forest"])

if selected_model == "Gradient Boosting":
    # Training van het Gradient Boosting Classifier model
    gradient_boosting_model = GradientBoostingClassifier()
    gradient_boosting_model.fit(X_resampled, y_resampled)
    gradient_boosting_predictions = gradient_boosting_model.predict(X_test)
    accuracy = accuracy_score(y_test, gradient_boosting_predictions) * 100
    st.write("Gradient Boosting Accuracy:", accuracy, '%')

    # Confusion Matrix
    cm = confusion_matrix(y_test, gradient_boosting_predictions)
    st.write("Confusion Matrix:")
    st.write(cm)
    st.pyplot(plt.figure(figsize=(8, 6)))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Voorspelde waarden')
    plt.ylabel('Werkelijke waarden')
    plt.title('Confusion Matrix - Gradient Boosting model')

elif selected_model == "Support Vector Machine (SVM)":
    # Training van het Support Vector Machine (SVM) model
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_resampled, y_resampled)
    svm_predictions = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, svm_predictions) * 100
    st.write("Support Vector Machine Accuracy:", accuracy, '%')

    # Confusion Matrix
    cm = confusion_matrix(y_test, svm_predictions)
    st.write("Confusion Matrix:")
    st.write(cm)
    st.pyplot(plt.figure(figsize=(8, 6)))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Voorspelde waarden')
    plt.ylabel('Werkelijke waarden')
    plt.title('Confusion Matrix - SVM model')

elif selected_model == "Random Forest":
    # Training van het Random Forest Classifier model
    random_forest_model = RandomForestClassifier()
    random_forest_model.fit(X_resampled, y_resampled)
    random_forest_predictions = random_forest_model.predict(X_test)
    accuracy = accuracy_score(y_test, random_forest_predictions) * 100
    st.write("Random Forest Accuracy:", accuracy, '%')

    # Confusion Matrix
    cm = confusion_matrix(y_test, random_forest_predictions)
    st.write("Confusion Matrix:")
    st.write(cm)
    st.pyplot(plt.figure(figsize=(8, 6)))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Voorspelde waarden')
    plt.ylabel('Werkelijke waarden')
    plt.title('Confusion Matrix - Random Forest Model')
