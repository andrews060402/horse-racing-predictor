import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

attributes = ['age', 'weight', 'sex', 'start_position', 'weather', 'track_conditions', 'finish_position']
# Step 1: Load the CSV file into a pandas DataFrame
df = pd.read_csv('../../data/woodbine_horses.csv', usecols=attributes)

# Assuming 'sex', 'weather', and 'track_conditions' are categorical
categorical_attributes = ['sex', 'weather', 'track_conditions']
label_encoders = {}

for cat_attr in categorical_attributes:
    label_encoder = LabelEncoder()
    df[cat_attr] = label_encoder.fit_transform(df[cat_attr])
    label_encoders[cat_attr] = label_encoder

# Save label encoders if you need to invert the transformation later

# Feature Scaling
scaler = StandardScaler()
numerical_attributes = ['age', 'weight', 'start_position']  # Update this list as needed
df[numerical_attributes] = scaler.fit_transform(df[numerical_attributes])

# Splitting the Data into Training and Testing Sets
# Assuming you have a target variable named 'outcome'
X = df.drop('finish_position', axis=1)
y = df['finish_position']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# MODEL TRAINING AND EVALUATION
# Initialize models
svm = SVC()
mlp = MLPClassifier()
logistic = LogisticRegression()

# Train models
svm.fit(X_train, y_train)
mlp.fit(X_train, y_train)
logistic.fit(X_train, y_train)

# Make predictions
svm_preds = svm.predict(X_test)
mlp_preds = mlp.predict(X_test)
logistic_preds = logistic.predict(X_test)

# Evaluate models
svm_f1 = f1_score(y_test, svm_preds)
mlp_f1 = f1_score(y_test, mlp_preds)
logistic_f1 = f1_score(y_test, logistic_preds)

# Print F1 scores
print(f'SVM F1 Score: {svm_f1}')
print(f'MLP F1 Score: {mlp_f1}')
print(f'Logistic Regression F1 Score: {logistic_f1}')
