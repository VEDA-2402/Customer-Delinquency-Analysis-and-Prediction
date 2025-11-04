import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

#combined_cleaned_data_df = pd.read_csv('combined_cleaned_data')
#print(combined_cleaned_data_df.head())-----> what i did wrong

# Read the cleaned csv file
cleaned_df = pd.read_csv('combined_cleaned_data.csv')

# Print the first five rows
#print(cleaned_df.head())

# Load the cleaned data
df = pd.read_csv('combined_cleaned_data.csv')

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

#===================CONCLUSION====================#
'''
After applying SMOTE to balance the training data, the XGBoost classifier achieved:
- Accuracy: 93.27%
- ROC AUC Score: 0.5254
- Precision (Default Class): 0.13
- Recall (Default Class): 0.02
- F1-Score (Default Class): 0.03

KEY FINDINGS:
-------------
1. CLASS IMBALANCE IMPACT:
   The high accuracy (93.27%) is misleading due to severe class imbalance in the 
   test set (~93.8% non-defaults). The model achieves high accuracy primarily by 
   correctly predicting the majority class (non-defaults).

2. MINORITY CLASS DETECTION CHALLENGE:
   Despite SMOTE oversampling, the model struggles to identify actual defaulters:
   - Only 2 out of 125 defaults were correctly identified (1.6% detection rate)
   - 123 defaults were missed (false negatives)
   This indicates the current feature set may not sufficiently capture patterns 
   that distinguish defaulters from non-defaulters.

3. MODEL DISCRIMINATION:
   ROC AUC of 0.5254 indicates the model has limited discriminative power, 
   performing only slightly better than random guessing (0.5).

BUSINESS IMPLICATIONS:
----------------------
- Missing 98.4% of defaults represents significant risk exposure for financial institutions
- The cost of false negatives (missed defaults) far outweighs false positives in credit risk
- Current model is NOT suitable for production deployment without significant improvements

RECOMMENDATIONS FOR IMPROVEMENT:
--------------------------------
1. Feature Engineering:
   - Add payment history patterns (rolling averages, trends)
   - Include credit utilization ratios over time
   - Incorporate external credit bureau data
   - Create interaction features between key predictors

2. Model Enhancement:
   - Perform hyperparameter tuning (GridSearchCV/RandomizedSearchCV)
   - Try ensemble methods with different algorithms
   - Implement cost-sensitive learning to penalize false negatives more heavily
   - Adjust classification threshold for business-optimized predictions

3. Data Collection:
   - Gather more balanced historical data
   - Include behavioral features (payment timing, utilization patterns)
   - Add macroeconomic indicators if available

4. Alternative Approaches:
   - Test Random Forest, LightGBM, or neural networks
   - Consider two-stage models (screening + detailed assessment)
   - Implement SHAP/LIME for model explainability
'''

