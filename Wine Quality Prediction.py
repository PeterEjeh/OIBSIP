import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import joblib

# Load the dataset
file_path = 'WineQT.csv'
wine_data = pd.read_csv(file_path)

# Drop unnecessary columns (e.g., 'Id')
wine_data = wine_data.drop(columns=['Id'], errors='ignore')

# Check for missing values
if wine_data.isnull().sum().sum() > 0:
    wine_data = wine_data.dropna()

# Analyze features like density and acidity
sns.pairplot(wine_data, vars=['density', 'pH', 'fixed acidity', 'volatile acidity'], hue='quality')
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(wine_data.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

# Split data into features and target variable
X = wine_data.drop('quality', axis=1)
y = wine_data['quality']

# Compute class weights to handle class imbalance
classes = np.unique(y)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
class_weights_dict = {cls: weight for cls, weight in zip(classes, class_weights)}

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define Pipelines
rf_pipeline = Pipeline([
    ('rf', RandomForestClassifier(random_state=42, class_weight='balanced'))
])

sgd_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('sgd', SGDClassifier(random_state=42, max_iter=1000, tol=1e-3, class_weight='balanced'))
])

svc_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(random_state=42, class_weight='balanced'))
])

# Random Forest Classifier with Grid Search
rf_params = {'rf__n_estimators': [50, 100, 200], 'rf__max_depth': [None, 10, 20], 'rf__min_samples_split': [2, 5, 10]}
rf_model = GridSearchCV(rf_pipeline, rf_params, cv=5)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.best_estimator_.predict(X_test)
print("Random Forest Classifier (Optimized):\n")
print(classification_report(y_test, y_pred_rf))

# SGD Classifier with Cross-Validation
cv_scores_sgd = cross_val_score(sgd_pipeline, X_train, y_train, cv=5, scoring='accuracy')
sgd_pipeline.fit(X_train, y_train)
y_pred_sgd = sgd_pipeline.predict(X_test)
print("SGD Classifier:\n")
print(f"Cross-Validation Accuracy: {np.mean(cv_scores_sgd):.4f}")
print(classification_report(y_test, y_pred_sgd))

# Support Vector Classifier (SVC) with Grid Search
svc_params = {'svc__C': [0.1, 1, 10], 'svc__kernel': ['linear', 'rbf']}
svc_model = GridSearchCV(svc_pipeline, svc_params, cv=5)
svc_model.fit(X_train, y_train)
y_pred_svc = svc_model.best_estimator_.predict(X_test)
print("Support Vector Classifier (Optimized):\n")
print(classification_report(y_test, y_pred_svc))

# Visualize feature importance for Random Forest
rf_best = rf_model.best_estimator_.named_steps['rf']
feature_importances = rf_best.feature_importances_
feature_names = X.columns
plt.figure(figsize=(10, 6))
plt.barh(feature_names, feature_importances, color='skyblue')
plt.title('Feature Importance (Random Forest)')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.tight_layout()
plt.show()

# Visualize confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
models = [(y_pred_rf, 'Random Forest'), (y_pred_sgd, 'SGD'), (y_pred_svc, 'SVC')]
for ax, (y_pred, title) in zip(axes, models):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f'{title} Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
plt.tight_layout()
plt.show()

# Save the best models
joblib.dump(rf_model.best_estimator_, 'random_forest_model.pkl')
joblib.dump(sgd_pipeline, 'sgd_model.pkl')
joblib.dump(svc_model.best_estimator_, 'svc_model.pkl')
