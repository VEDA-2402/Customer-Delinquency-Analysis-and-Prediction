import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')


# Read the cleaned csv file
cleaned_df = pd.read_csv('combined_cleaned_data.csv')

# Print the first five rows
print(cleaned_df.head())

# Load the cleaned data
df = pd.read_csv('combined_cleaned_data.csv')

#==================================================================#
#MODEL TRAINING
#==================================================================#

# Define target variable (adjust the column name as per your dataset)
y = df['Delinquent_Acc']  # assuming this is your target column

# Define feature variables by dropping the target column
X = df.drop(columns=['Delinquent_Acc'])

# Split data into train and test sets (adjust test_size as needed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit scaler on training data and transform both train and test data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Optional: convert scaled arrays back to dataframes with feature names for easier handling
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

# Verify the result
#print(X_train_scaled.head())
#print(X_test_scaled.head())

from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# Initialize the XGBoost classifier
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Train the model
xgb_model.fit(X_train_scaled, y_train)

# Predict on test data
y_pred = xgb_model.predict(X_test_scaled)
y_pred_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]  # probability of positive class

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.4f}')
print(f'ROC AUC Score: {roc_auc:.4f}')
print('Classification Report:')
print(report)

print(y_train.value_counts(), y_test.value_counts())
# due to the polarization of the values the high accuracy is misleading 
# resampling the training set using SMOTE

#=================================================================================#
#SMOTE
#=================================================================================#

from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# Apply SMOTE only to training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Check the new class balance
print(y_train_resampled.value_counts())

# Train model on resampled data
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train_resampled, y_train_resampled)

# Predict on (unmodified) test set
y_pred = xgb_model.predict(X_test_scaled)
y_pred_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]

# Show evaluation metrics for new model
print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
print(f'ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}')
print('Classification Report:')
print(classification_report(y_test, y_pred))

#===============VISUALS AND ANALYSIS=================#

#CONFUSION MATRIX

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

#ROE CURVE AND PRECISION-RECALL CURVE

from sklearn.metrics import roc_curve, auc, precision_recall_curve

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

#FEATURE IMPORTANCE 
from xgboost import plot_importance
plot_importance(xgb_model, max_num_features=10)
plt.title('Feature Importance')
plt.show()




