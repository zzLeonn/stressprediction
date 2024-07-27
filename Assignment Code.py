import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.naive_bayes import GaussianNB 
from sklearn.neighbors import KNeighborsClassifier
import plotly.graph_objects as go

# Specify the directory path
directory_path = "D:/school/BSC Computer Science/CET313 Artificial Intelligence/Assignment"

# Create a list of file names by iterating over the contents of the directory
file_list = [file.name for file in Path(directory_path).iterdir() if file.is_file()]

# Read CSV
data_set = pd.read_csv("D:/school/BSC Computer Science/CET313 Artificial Intelligence/Assignment/sleephealthlifestyle.csv")

# Display last 5 rows of dataset
print(data_set.tail(5))

# Get and display the column names of the dataset
print(data_set.columns)

# Get information about the dataset and print
info = data_set.info()
print(info)

# Statistical summary
stat = data_set.describe().T
print(stat)

# Calculate the number of missing values in dataset
snull = data_set.isnull().sum()

# Print the count of missing values
print(snull)

rows = data_set[pd.isna(data_set["Sleep Disorder"])]
print(rows)
stress = data_set.fillna("Nothing")
print(data_set.head(1))

#since both Normal Weight and Normal is the same, Replace it with just one Normal
data_set["BMI Category"] = data_set["BMI Category"].replace("Normal Weight", "Normal")

# Using person id and gender to check for same ID but different gender and drop those data
no_duplicates = data_set.drop_duplicates(subset=['Person ID', 'Gender'])
print(no_duplicates['Stress Level'].value_counts())

# Calculate the count of values in the 'Stress Level' column
stress_count = data_set["Stress Level"].count()

# Calculate the mean (average) of values in the 'Stress Level' column
stress_mean = data_set["Stress Level"].mean()

# Print the count and mean of the 'stress' column to the console
print("Count:", stress_count)
print("Mean:", stress_mean)

# The dataset includes categorical data, such as "Occupation", "BMI Category", and "Sleep Disorder Columns".
# 'Label Encoding' to convert these categorical values into numerical data.
label_encoder = LabelEncoder()
cat_cols = ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']
for col in cat_cols:
    data_set[col] = label_encoder.fit_transform(data_set[col])
print(data_set)

# Splitting Blood pressure column into 2 other columns as the format does not work
data_set[['Systolic BP', 'Diastolic BP']] = data_set['Blood Pressure'].str.split('/', expand=True)
    
# Convert the new columns to numeric type
data_set[['Systolic BP', 'Diastolic BP']] = data_set[['Systolic BP', 'Diastolic BP']].apply(pd.to_numeric)
    
# Drop the original 'Blood Pressure' column
data_set = data_set.drop('Blood Pressure', axis=1)

data_set.head(1)

# Drop 'Person ID' column
data_set.drop('Person ID', axis=1, inplace=True)

# Moving Stress Level column
stress_level_index = data_set.columns.get_loc('Stress Level')
columns = list(data_set.columns[:stress_level_index]) + list(data_set.columns[stress_level_index + 1:]) + ['Stress Level']
data_set = data_set[columns]
print(data_set.tail(1))

# Data visualization
# Histograms for each numerical feature
data_set.hist(figsize=(10, 8))
plt.suptitle("Histograms of Numerical Features", fontsize=16)
plt.show()

# Keep only the 8 features you want to use
selected_features = ['Gender', 'Age', 'Occupation', 'Sleep Duration', 'BMI Category', 'Heart Rate', 'Daily Steps', 'Systolic BP']
X = data_set[selected_features]
y = data_set['Stress Level']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression with increased max_iter
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train_scaled, y_train)

# Predicting the values
predicted = logistic_model.predict(X_test_scaled)
# Confusion matrix
conf = confusion_matrix(y_test, predicted)
print("Confusion Matrix : \n", conf)
cr = classification_report(y_test, predicted)
print("Classification Report:\n", cr)
# Printing the test accuracy
print("The test accuracy of Logistic Regression is : ", accuracy_score(y_test, predicted) * 100, "%")

# Naive Bayes
naive_bayes = GaussianNB()
naive_bayes.fit(X_train, y_train)
y_predict = naive_bayes.predict(X_test)
matrix = confusion_matrix(y_test, y_predict)
report = classification_report(y_test, y_predict)
accuracy_nb = accuracy_score(y_test, y_predict)
print("Naive Bayes:")
print("Confusion Matrix:")
print(matrix)
print("Classification Report:")
print(report)
print("The test accuracy of Naive Bayes is:", accuracy_nb * 100, "%")

# KNN Classifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_predict_knn = knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_predict_knn)
conf_matrix_knn = confusion_matrix(y_test, y_predict_knn)
classification_rep_knn = classification_report(y_test, y_predict_knn)
print("KNN Classifier:")
print("Accuracy:", accuracy_knn)
print("Confusion Matrix:")
print(conf_matrix_knn)
print("Classification Report:")
print(classification_rep_knn)
print("The test accuracy of KNN Classifier is:", accuracy_knn * 100, "%")

# Define the model names and their corresponding accuracy scores
model_names = ['Logistic Regression', 'Naive Bayes', 'KNN']
accuracy_scores = [89, 90, 93]

# Create a color map for each model
color_map = {
    'Logistic Regression': 'rgb(255, 127, 14)',
    'Naive Bayes': 'rgb(44, 160, 44)',
    'KNN': 'rgb(31, 119, 180)'
}

# Create a trace for each model
traces = []
for model, accuracy, color in zip(model_names, accuracy_scores, [color_map[model] for model in model_names]):
    trace = go.Bar(
        x=[model],
        y=[accuracy],
        text=[f'{accuracy:.2f}%'],
        name=model,
        marker=dict(color=color),
        textposition='outside'
    )
    traces.append(trace)

# Create the layout
layout = go.Layout(
    title='Comparison of Model Accuracy',
    xaxis=dict(title='Model'),
    yaxis=dict(title='Accuracy Score'),
    width=700,
    height=600
)

# Create the figure
fig = go.Figure(data=traces, layout=layout)

# Show the plot
fig.show()

def get_user_input():
    gender = int(input("Gender (Male: 1, Female: 0): "))
    age = int(input("Age: "))
    print("['Scientist = 0', 'Doctor = 1', 'Accountant = 2', 'Teacher = 3', 'Manager = 4', 'Engineer = 5', 'Sales Representative = 6', 'Lawyer = 8', 'Salesperson = 7', 'Software Engineer = 9', 'Nurse = 10']")   
    occupation = int(input("Occupation (encoded): "))
    sleep_duration = float(input("Sleep Duration (hours): "))
    bmi_category = int(input("BMI Category (Underweight: 1, Normal: 2, Overweight: 3): "))
    heart_rate = int(input("Heart Rate (bpm): "))
    daily_steps = int(input("Daily Steps: "))
    systolic_bp = int(input("Systolic Blood Pressure: "))
    
    return np.array([[gender, age, occupation, sleep_duration, bmi_category, heart_rate, daily_steps, systolic_bp]])

predicted_stress = knn.predict(get_user_input())
print("Predicted Stress Level using KNN Classifier:", predicted_stress)
