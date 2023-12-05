import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

attributes = ['age', 'weight', 'sex', 'start_position', 'weather', 'track_conditions', 'finish_position']

# Step 1: Load the CSV file into a pandas DataFrame
df = pd.read_csv('../../data/woodbine_horses.csv', usecols=attributes)

# Exclude rows where 'age' or 'weight' are zero
df = df[(df['age'] != 0) & (df['weight'] != 0)]

df['top_3_finish'] = df['finish_position'].apply(lambda x: 1 if x in [1, 2, 3] else 0)

# Assuming 'sex', 'weather', and 'track_conditions' are categorical
categorical_attributes = ['sex', 'weather', 'track_conditions']
label_encoders = {}

for cat_attr in categorical_attributes:
    label_encoder = LabelEncoder()
    df[cat_attr] = label_encoder.fit_transform(df[cat_attr])
    label_encoders[cat_attr] = label_encoder

# Feature Scaling
scaler = StandardScaler()
numerical_attributes = ['age', 'weight', 'start_position']
df[numerical_attributes] = scaler.fit_transform(df[numerical_attributes])

# Splitting the Data into Training and Testing Sets
X = df.drop(['finish_position', 'top_3_finish'], axis=1)
y = df['top_3_finish']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# MODEL TRAINING AND EVALUATION
# Initialize models
svm = SVC(C=10, gamma='scale', probability=True)  # Added probability=True for SVM to output probabilities
mlp = MLPClassifier(max_iter=500, learning_rate_init=0.001)
logistic = LogisticRegression(solver='liblinear', max_iter=500, penalty='l1', C=0.5)

# Train models
svm.fit(X_train, y_train)
mlp.fit(X_train, y_train)
logistic.fit(X_train, y_train)

# Make predictions
svm_preds = svm.predict(X_test)
mlp_preds = mlp.predict(X_test)
logistic_preds = logistic.predict(X_test)

# Evaluate models
svm_f1 = f1_score(y_test, svm_preds, average='weighted')
mlp_f1 = f1_score(y_test, mlp_preds, average='weighted')
logistic_f1 = f1_score(y_test, logistic_preds, average='weighted')

# Print F1 scores
print(f'SVM F1 Score: {svm_f1}')
print(f'MLP F1 Score: {mlp_f1}')
print(f'Logistic Regression F1 Score: {logistic_f1}')


# Function to calculate TP, FP, FN
def calculate_tp_fp_fn(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp, fp, fn

# Evaluate models and calculate TP, FP, FN
for model_name, model in zip(['SVM', 'MLP', 'Logistic Regression'], [svm, mlp, logistic]):
    preds = model.predict(X_test)
    tp, fp, fn = calculate_tp_fp_fn(y_test, preds)
    print(f"{model_name} - True Positives: {tp}, False Positives: {fp}, False Negatives: {fn}")

# Define a function to adjust predictions based on a new threshold
def adjust_predictions(model, X, threshold=0.5):
    probabilities = model.predict_proba(X)[:, 1]  # Probabilities of the positive class
    return np.where(probabilities >= threshold, 1, 0)

# Adjust threshold and recalculate metrics for each model
new_threshold = 0.4  # Example threshold, can be tuned

for model_name, model in zip(['SVM', 'MLP', 'Logistic Regression'], [svm, mlp, logistic]):
    # Adjust predictions based on the new threshold
    adjusted_preds = adjust_predictions(model, X_test, new_threshold)

    # Recalculate metrics
    tp, fp, fn = calculate_tp_fp_fn(y_test, adjusted_preds)
    adjusted_f1 = f1_score(y_test, adjusted_preds, average='weighted')
    adjusted_precision = precision_score(y_test, adjusted_preds, average='weighted')
    adjusted_recall = recall_score(y_test, adjusted_preds, average='weighted')

    # Print the recalculated metrics
    print(f"{model_name} with threshold {new_threshold}:")
    print(f"True Positives: {tp}, False Positives: {fp}, False Negatives: {fn}")
    print(f"Adjusted Precision: {adjusted_precision}")
    print(f"Adjusted Recall: {adjusted_recall}")
    print(f"Adjusted F1 Score: {adjusted_f1}\n")
