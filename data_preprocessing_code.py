# ===============================
# DATA PREPROCESSING PIPELINE
# ===============================

import pandas as pd
import numpy as np

# ===============================
# Data Collection
# ===============================
df = pd.read_csv("dataset.csv")

print("Before cleaning:\n")
print(df.head())

# ===============================
# Data Understanding
# ===============================
print(df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())

# ===============================
# Duplicate Handling
# ===============================
df.duplicated().sum()

df = df.drop_duplicates()
print("After Removed duplicates\n:", df)

# ===============================
# Column Name Cleaning
# ===============================
df.columns = df.columns.str.strip()

# ===============================
# Handling Missing Values
# ===============================
df = df.replace(r'^\s*$', np.nan, regex=True)

# Fill numeric
 df['transaction_amount'] = df['transaction_amount'].fillna(df['transaction_amount'].median())
 df['account_balance'] = df['account_balance'].fillna(df['account_balance'].mean())

# Fill categorical
cols = ['location', 'transaction_type', 'merchant_category', 'device_type']
for col in cols:
    df[col] = df[col].fillna("Unknown")

print("\n Missing Values Handled")
print(df.isnull().sum())

# ===============================
# Data Type Conversion
# ===============================
df['transaction_id'] = df['transaction_id'].astype(int)
df['customer_id'] = df['customer_id'].astype(int)
df['account_balance'] = df['account_balance'].round().astype(int)

print("\n Data Types Fixed")
print(df.dtypes)

# ===============================
# Text Standardization
# ===============================
df['transaction_type'] = df['transaction_type'].str.capitalize()
df['merchant_category'] = df['merchant_category'].str.capitalize()
df['location'] = df['location'].str.capitalize()

df['device_type'] = df['device_type'].apply(
    lambda x: "ATM" if str(x).upper() == "ATM" else str(x).capitalize()
)

print("\n Text Standardized")
print(df.head(10))

# ===============================
# Invalid Data Removal
# ===============================
df = df[df['transaction_amount'] > 0]
df = df[df['account_balance'] >= 0]

print("\n Invalid Data Removed")
print("Shape:", df.shape)

# ===============================
# Outlier Detection (IQR)
# ===============================
Q1 = df['transaction_amount'].quantile(0.25)
Q3 = df['transaction_amount'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

before = df.shape[0]

df = df[(df['transaction_amount'] >= lower) & (df['transaction_amount'] <= upper)]

after = df.shape[0]

print("Rows before outlier removal:", before)
print("Rows after outlier removal:", after)

# ===============================
# Feature Engineering
# ===============================
df['transaction_time'] = pd.to_datetime(
    df['transaction_time'],
    format='%d-%m-%Y %H:%M',
    errors='coerce'
)

df['transaction_Date'] = df['transaction_time'].dt.strftime('%d-%m-%Y')

df['transaction_hour'] = df['transaction_time'].dt.hour
df['transaction_day'] = df['transaction_time'].dt.day
df['transaction_month'] = df['transaction_time'].dt.month

print(df[['transaction_time', 'transaction_Date',
          'transaction_hour', 'transaction_day', 'transaction_month']].head())

# ===============================
# Sorting
# ===============================
df = df.sort_values(by='transaction_id')
print("\n Data Sorted")

# ===============================
# Save Clean Dataset
# ===============================
print("Final Shape:", df.shape)
df.to_csv("cleaned_dataset.csv", index=False)
print("Clean dataset saved successfully!")

# ===============================
# Encoding
# ===============================
df = pd.get_dummies(df, columns=[
    'transaction_type', 'merchant_category', 'location', 'device_type'
], drop_first=True)

df[df.select_dtypes(include='bool').columns] = df.select_dtypes(include='bool').astype(int)

print("\nAfter Encoding Shape:", df.shape)
print("\nColumns After Encoding:")
print(df.columns)
print("\nSample Data After Encoding:")
print(df.head())

# ===============================
# Scaling
# ===============================
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df[['transaction_amount', 'account_balance']] = scaler.fit_transform(
    df[['transaction_amount', 'account_balance']]
)

print("\nScaling Completed ")
print("\nSample Data After Scaling:")
print(df[['transaction_amount', 'account_balance']].head())

# ===============================
# Final Check
# ===============================
print("\nFinal Dataset Shape:", df.shape)
print("\nFinal Data Preview:")
print(df.head())

# ===============================
# Feature Selection
# ===============================
df = df.drop(columns=['transaction_time'])
print(df.tail())

# ===============================
# Save Final Dataset
# ===============================
print("Final Shape:", df.shape)
df.to_csv("final_dataset.csv", index=False)
print("Final dataset saved successfully!")