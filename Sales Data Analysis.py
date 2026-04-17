import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# Load Dataset
# =========================
df = pd.read_csv(r"D:\Sales Data Analysis\sales_data.csv")

# =========================
# Basic Data Inspection
# =========================
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

# =========================
# Data Cleaning
# =========================

# Remove duplicates
df.drop_duplicates(inplace=True)

# Convert date column
df['Sale_Date'] = pd.to_datetime(df['Sale_Date'])

# Remove invalid values
df = df[df['Sales_Amount'] > 0]
df = df[df['Quantity_Sold'] > 0]

# Fill missing values
df.bfill(inplace=True)

# =========================
# Feature Engineering
# =========================

# Profit Calculation
df['Profit'] = (df['Unit_Price'] - df['Unit_Cost']) * df['Quantity_Sold']

# Extract Month & Year
df['Month'] = df['Sale_Date'].dt.month
df['Year'] = df['Sale_Date'].dt.year

print(df['Profit'].head())
print(df['Month'].head())
print(df['Year'].head())

# =========================
# Exploratory Data Analysis (EDA)
# =========================

region_sales = df.groupby('Region')['Sales_Amount'].sum()
category_sales = df.groupby('Product_Category')['Sales_Amount'].sum()
rep_sales = df.groupby('Sales_Rep')['Sales_Amount'].sum()
customer_sales = df.groupby('Customer_Type')['Sales_Amount'].sum()
monthly_sales = df.groupby('Month')['Sales_Amount'].sum()

print(region_sales)
print(category_sales)
print(rep_sales)
print(customer_sales)
print(monthly_sales)

# =========================
# Data Visualization
# =========================

# Sales by Region 
region_sales.plot(kind='bar')
plt.title("Sales by Region")
plt.xlabel("Region")
plt.ylabel("Sales Amount")
plt.savefig("images/sales_by_region.png", bbox_inches='tight')
plt.show()

#Product Category Sales
sns.barplot(x=category_sales.index, y=category_sales.values)
plt.title("Sales by Product Category")
plt.savefig("images/category_sales.png", bbox_inches='tight')
plt.show()

#Monthly Trend
monthly_sales.plot(kind='line')
plt.title("Monthly Sales Trend")
plt.savefig("images/monthly_trend.png", bbox_inches='tight')
plt.show()

# Profit Distribution 
sns.histplot(df['Profit'])
plt.title("Profit Distribution")
plt.savefig("images/profit_distribution.png", bbox_inches='tight')
plt.show()