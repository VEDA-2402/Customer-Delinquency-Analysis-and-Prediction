import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# Load data
customer_df = pd.read_csv('customer.csv')
credit_card_df = pd.read_csv('credit_card.csv')
merged_df = customer_df.merge(credit_card_df, on='Client_Num', how='inner')

print('Delinquency = When a credit card customer fails to make their minimum payment for 30+ days past the due date.')

print("="*80)
print("DELINQUENCY ANALYSIS BY DEMOGRAPHICS")
print("="*80)

# Question 1: Delinquency by Age Group
print("\n1. DELINQUENCY BY AGE GROUP")
print("-" * 50)
merged_df['Age_Group'] = pd.cut(merged_df['Customer_Age'], 
                                 bins=[20, 30, 40, 50, 60, 100], 
                                 labels=['20-30', '30-40', '40-50', '50-60', '60+'])

age_delinq = merged_df.groupby('Age_Group')['Delinquent_Acc'].agg(['count', 'sum', 'mean'])
age_delinq.columns = ['Total_Customers', 'Delinquent_Count', 'Delinquency_Rate']
age_delinq['Delinquency_Rate'] = age_delinq['Delinquency_Rate'] * 100
print(age_delinq)

# Question 2: Delinquency by Gender
print("\n2. DELINQUENCY BY GENDER")
print("-" * 50)
gender_delinq = merged_df.groupby('Gender')['Delinquent_Acc'].agg(['count', 'sum', 'mean'])
gender_delinq.columns = ['Total_Customers', 'Delinquent_Count', 'Delinquency_Rate']
gender_delinq['Delinquency_Rate'] = gender_delinq['Delinquency_Rate'] * 100
print(gender_delinq)

# Question 3: Delinquency by Job Category
print("\n3. DELINQUENCY BY JOB CATEGORY")
print("-" * 50)
job_delinq = merged_df.groupby('Customer_Job')['Delinquent_Acc'].agg(['count', 'sum', 'mean'])
job_delinq.columns = ['Total_Customers', 'Delinquent_Count', 'Delinquency_Rate']
job_delinq['Delinquency_Rate'] = job_delinq['Delinquency_Rate'] * 100
job_delinq = job_delinq.sort_values('Delinquency_Rate', ascending=False)
print(job_delinq)

# Question 4: Delinquency by Income Level
print("\n4. DELINQUENCY BY INCOME LEVEL")
print("-" * 50)
merged_df['Income_Group'] = pd.cut(merged_df['Income'], 
                                    bins=[0, 40000, 80000, 120000, 160000, 300000],
                                    labels=['<40K', '40K-80K', '80K-120K', '120K-160K', '160K+'])

income_delinq = merged_df.groupby('Income_Group')['Delinquent_Acc'].agg(['count', 'sum', 'mean'])
income_delinq.columns = ['Total_Customers', 'Delinquent_Count', 'Delinquency_Rate']
income_delinq['Delinquency_Rate'] = income_delinq['Delinquency_Rate'] * 100
print(income_delinq)

# Question 5: Delinquency by Satisfaction Score
print("\n5. DO SATISFIED CUSTOMERS DEFAULT LESS?")
print("-" * 50)
satisfaction_delinq = merged_df.groupby('Cust_Satisfaction_Score')['Delinquent_Acc'].agg(['count', 'sum', 'mean'])
satisfaction_delinq.columns = ['Total_Customers', 'Delinquent_Count', 'Delinquency_Rate']
satisfaction_delinq['Delinquency_Rate'] = satisfaction_delinq['Delinquency_Rate'] * 100
print(satisfaction_delinq)

# Question 6: Delinquency by Marital Status
print("\n6. DELINQUENCY BY MARITAL STATUS")
print("-" * 50)
marital_delinq = merged_df.groupby('Marital_Status')['Delinquent_Acc'].agg(['count', 'sum', 'mean'])
marital_delinq.columns = ['Total_Customers', 'Delinquent_Count', 'Delinquency_Rate']
marital_delinq['Delinquency_Rate'] = marital_delinq['Delinquency_Rate'] * 100
print(marital_delinq)

# Question 7: Delinquency by Number of Dependents
print("\n7. DELINQUENCY BY NUMBER OF DEPENDENTS")
print("-" * 50)
dependent_delinq = merged_df.groupby('Dependent_Count')['Delinquent_Acc'].agg(['count', 'sum', 'mean'])
dependent_delinq.columns = ['Total_Customers', 'Delinquent_Count', 'Delinquency_Rate']
dependent_delinq['Delinquency_Rate'] = dependent_delinq['Delinquency_Rate'] * 100
print(dependent_delinq)

# Summary - Which factor has strongest correlation with delinquency
print("\n" + "="*80)
print("SUMMARY: KEY INSIGHTS")
print("="*80)

print("\nHighest Delinquency Rates by Factor:")
print(f"Age Group: {age_delinq['Delinquency_Rate'].idxmax()} ({age_delinq['Delinquency_Rate'].max():.2f}%)")
print(f"Job Category: {job_delinq['Delinquency_Rate'].idxmax()} ({job_delinq['Delinquency_Rate'].max():.2f}%)")
print(f"Income Group: {income_delinq['Delinquency_Rate'].idxmax()} ({income_delinq['Delinquency_Rate'].max():.2f}%)")
print(f"Satisfaction: Score {satisfaction_delinq['Delinquency_Rate'].idxmax()} ({satisfaction_delinq['Delinquency_Rate'].max():.2f}%)")

# Calculate correlation strength
print("\nCorrelation with Delinquency:")
correlation = merged_df[['Customer_Age', 'Income', 'Cust_Satisfaction_Score', 'Dependent_Count', 'Delinquent_Acc']].corr()['Delinquent_Acc'].sort_values(ascending=False)
print(correlation)

print("\n" + "="*80)
