# Import libraries
import numpy as np
import pandas as pd


# Read the CSV files
customer_df = pd.read_csv('customer.csv')
credit_card_df = pd.read_csv('credit_card.csv')

# Merge the dataframes on client_ID
combined_df = pd.merge(customer_df, credit_card_df, on='Client_Num', how='inner')

#Droping the columns which are not relevant to train the model 
# List of columns to drop
cols_to_drop = [
    'Card_Category', 'Week_Start_Date', 'Week_Num', 'Qtr', 'current_year',
    'Gender', 'Education_Level', 'Marital_Status', 'state_cd', 'Zipcode',
    'contact', 'Customer_Job', 'Use Chip', 'Exp Type'
]

# Drop the columns from the combined dataframe
combined_df = combined_df.drop(columns=cols_to_drop)

# Display the remaining columns after dropping
print("Remaining columns after drop:")
print(combined_df.columns)

# Check data types of all columns in combined_df
#print(combined_df.dtypes)

#changing the columns with object data type 
combined_df['Personal_loan'] = combined_df['Personal_loan'].map({'yes': 1, 'no': 0})
combined_df['House_Owner'] = combined_df['House_Owner'].map({'yes': 1, 'no': 0})
combined_df['Car_Owner'] = combined_df['Car_Owner'].map({'yes': 1, 'no': 0})

print(combined_df.dtypes)

# Check the count of missing values per column
#-->missing_values = combined_df.isnull().sum()

# Display columns with missing values and their counts
#-->print(missing_values[missing_values > 0])
#No missing values 

# Save the cleaned combined dataframe to a new CSV file
combined_df.to_csv('combined_cleaned_data.csv', index=False)

print("Cleaned data saved to 'combined_cleaned_data.csv'")
