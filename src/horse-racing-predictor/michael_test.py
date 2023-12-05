import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
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

# Save label encoders if you need to invert the transformation later

# Feature Scaling
scaler = StandardScaler()
numerical_attributes = ['age', 'weight', 'start_position']
df[numerical_attributes] = scaler.fit_transform(df[numerical_attributes])

# Splitting the Data into Training and Testing Sets
# Assuming you have a target variable named 'outcome'
X = df.drop(['finish_position', 'top_3_finish'], axis=1)
y = df['top_3_finish']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# MODEL TRAINING AND EVALUATION
# Initialize models
svm = SVC(C=10, gamma='scale')
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

# Evaluate models and print results
for model_name, model in zip(['SVM', 'MLP', 'Logistic Regression'], [svm, mlp, logistic]):
    preds = model.predict(X_test)
    f1 = f1_score(y_test, preds, average='binary')
    precision = precision_score(y_test, preds, average='binary')
    recall = recall_score(y_test, preds, average='binary')
    f2 = 2 * (precision * recall) / (precision + recall)
    print(f"{model_name} - Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
    print(f2)

# Function to calculate TP, FP, FN
def calculate_tp_fp_fn(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp, fp, fn

# Evaluate models and calculate TP, FP, FN
for model_name, model in zip(['SVM', 'MLP', 'Logistic Regression'], [svm, mlp, logistic]):
    preds = model.predict(X_test)

    tp, fp, fn = calculate_tp_fp_fn(y_test, preds)
    print(f"{model_name} - True Positives: {tp}, False Positives: {fp}, False Negatives: {fn}")

# # Bar plot of F1 scores
# models = ['SVM', 'MLP', 'Logistic Regression']
# scores = [svm_f1, mlp_f1, logistic_f1]
# plt.bar(models, scores)
# plt.xlabel('Models')
# plt.ylabel('F1 Score')
# plt.title('Model Performance Comparison')
# plt.show()