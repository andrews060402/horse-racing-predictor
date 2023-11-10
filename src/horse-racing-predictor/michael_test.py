import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


attributes = ['age', 'weight', 'sex', 'start_position', 'weather', 'track_conditions']
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
X = df.drop('outcome', axis=1)  # Replace 'outcome' with your actual target column
y = df['outcome']  # Replace with actual target column

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
