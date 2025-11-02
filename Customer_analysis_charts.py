import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load customer data
customer_df = pd.read_csv('customer.csv')

# Set style
sns.set_style("whitegrid")

# Create figure with subplots
fig = plt.figure(figsize=(18, 12))

# 1. GENDER
ax1 = plt.subplot(3, 3, 1)
gender_counts = customer_df['Gender'].value_counts()
gender_counts.plot(kind='bar', color=['#FF6B9D', '#4A90E2'], ax=ax1, edgecolor='black')
ax1.set_title('Gender Distribution', fontsize=12, fontweight='bold')
ax1.set_ylabel('Count')
ax1.tick_params(axis='x', rotation=0)

# 2. AGE - Simple Histogram 
ax2 = plt.subplot(3, 3, 2)
ax2.hist(customer_df['Customer_Age'], bins=20, color='#4A90E2', edgecolor='black', alpha=0.7)
ax2.set_title('Age Distribution', fontsize=12, fontweight='bold')
ax2.set_xlabel('Age')
ax2.set_ylabel('Frequency')

# 3. EDUCATION
ax3 = plt.subplot(3, 3, 3)
education_counts = customer_df['Education_Level'].value_counts().sort_values()
education_counts.plot(kind='barh', color='#50C878', ax=ax3, edgecolor='black')
ax3.set_title('Education Level', fontsize=12, fontweight='bold')

# 4. MARITAL STATUS
ax4 = plt.subplot(3, 3, 4)
marital_counts = customer_df['Marital_Status'].value_counts()
ax4.pie(marital_counts, labels=marital_counts.index, autopct='%1.1f%%', 
        colors=['#FFD700', '#FF6B6B', '#95E1D3'], startangle=90)
ax4.set_title('Marital Status', fontsize=12, fontweight='bold')

# 5. DEPENDENT COUNT
ax5 = plt.subplot(3, 3, 5)
dependent_counts = customer_df['Dependent_Count'].value_counts().sort_index()
dependent_counts.plot(kind='bar', color='#9B59B6', ax=ax5, edgecolor='black')
ax5.set_title('Number of Dependents', fontsize=12, fontweight='bold')
ax5.set_ylabel('Count')
ax5.tick_params(axis='x', rotation=0)

# 6. INCOME
ax6 = plt.subplot(3, 3, 6)
ax6.hist(customer_df['Income'], bins=30, color='#E74C3C', edgecolor='black', alpha=0.7)
ax6.set_title('Income Distribution', fontsize=12, fontweight='bold')
ax6.set_xlabel('Income (₹)')
ax6.set_ylabel('Frequency')

# 7. JOB TYPE
ax7 = plt.subplot(3, 3, 7)
job_counts = customer_df['Customer_Job'].value_counts().sort_values()
job_counts.plot(kind='barh', color='#1ABC9C', ax=ax7, edgecolor='black')
ax7.set_title('Customer Job Distribution', fontsize=12, fontweight='bold')

# 8. SATISFACTION SCORE
ax8 = plt.subplot(3, 3, 8)
satisfaction_counts = customer_df['Cust_Satisfaction_Score'].value_counts().sort_index()
satisfaction_counts.plot(kind='bar', color=['#E74C3C', '#F39C12', '#F1C40F', '#2ECC71', '#27AE60'], ax=ax8, edgecolor='black')
ax8.set_title('Satisfaction Score (1-5)', fontsize=12, fontweight='bold')
ax8.set_ylabel('Count')
ax8.tick_params(axis='x', rotation=0)

# 9. ASSET OWNERSHIP
ax9 = plt.subplot(3, 3, 9)
car = customer_df['Car_Owner'].value_counts()
house = customer_df['House_Owner'].value_counts()
x = np.arange(2)
width = 0.35
ax9.bar(x - width/2, [car.get('no', 0), car.get('yes', 0)], width, label='Car', color='#3498DB', edgecolor='black')
ax9.bar(x + width/2, [house.get('no', 0), house.get('yes', 0)], width, label='House', color='#E67E22', edgecolor='black')
ax9.set_title('Asset Ownership', fontsize=12, fontweight='bold')
ax9.set_xticks(x)
ax9.set_xticklabels(['No', 'Yes'])
ax9.legend()

plt.tight_layout()
plt.savefig('demographic_analysis.png', dpi=300, bbox_inches='tight')
print("Charts saved as 'demographic_analysis.png'")
plt.show()
