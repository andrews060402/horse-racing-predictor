import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Attributes specified in your original code
attributes = ['age', 'weight', 'sex', 'start_position', 'weather', 'track_conditions', 'finish_position']

# Step 1: Load the CSV file into a pandas DataFrame
# Replace the file path with the correct path to your CSV file
df = pd.read_csv('../../data/woodbine_horses.csv', usecols=attributes)

# Exclude rows where 'age' or 'weight' are zero
df = df[(df['age'] != 0) & (df['weight'] != 0)]

# Create binary outcome for top 3 finish
df['top_3_finish'] = df['finish_position'].apply(lambda x: 1 if x in [1, 2, 3] else 0)

# Encoding categorical attributes
categorical_attributes = ['sex', 'weather', 'track_conditions']
for cat_attr in categorical_attributes:
    df[cat_attr] = LabelEncoder().fit_transform(df[cat_attr])

# Scaling numerical attributes
numerical_attributes = ['age', 'weight', 'start_position']
df[numerical_attributes] = StandardScaler().fit_transform(df[numerical_attributes])

# Splitting data
X = df.drop(['finish_position', 'top_3_finish'], axis=1)
y = df['top_3_finish']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Reducing the features to 2D for visualization
pca = PCA(n_components=2)
X_train_2d = pca.fit_transform(X_train)
X_test_2d = pca.transform(X_test)

# Train the SVM model on the 2D data
svm_2d = SVC(C=10, gamma='scale')
svm_2d.fit(X_train_2d, y_train)


# Function to plot the decision boundary
def plot_decision_boundary(clf, X, y, plot_support=True):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap='autumn')
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

    if plot_support:
        ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100, linewidth=1, facecolors='none',
                   edgecolors='k')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


# Visualize the SVM decision boundary on 2D data
plt.figure(figsize=(8, 6))
plot_decision_boundary(svm_2d, X_train_2d, y_train)
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
plt.title('SVM Decision Boundary with Support Vectors')
plt.show()

# [Continue with the rest of your original code]
