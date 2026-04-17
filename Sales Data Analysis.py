import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# Load Dataset
# =========================
df = pd.read_csv("sales_data.csv")

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
df['Week'] = df['Sale_Date'].dt.isocalendar().week

print(df['Profit'].head())
print(df['Month'].head())
print(df['Year'].head())
print(df['Week'].head())

# =========================
# Exploratory Data Analysis (EDA)
# =========================

region_sales = df.groupby('Region')['Sales_Amount'].sum()
category_sales = df.groupby('Product_Category')['Sales_Amount'].sum()
rep_sales = df.groupby('Sales_Rep')['Sales_Amount'].sum()
customer_sales = df.groupby('Customer_Type')['Sales_Amount'].sum()
monthly_sales = df.groupby('Month')['Sales_Amount'].sum()
weekly_sales = df.groupby('Week')['Sales_Amount'].sum()
monthly_sales_full = df.groupby(['Year', 'Month'])['Sales_Amount'].sum()

print(region_sales)
print(category_sales)
print(rep_sales)
print(customer_sales)
print(monthly_sales)
print(weekly_sales)
print(monthly_sales_full)

# =========================
# Profit & Loss Analysis
# =========================

total_profit = df[df['Profit'] > 0]['Profit'].sum()
loss_df = df[df['Profit'] < 0]
total_loss = loss_df['Profit'].sum()

print("Total Profit:", total_profit)
print("Total Loss:", total_loss)

# =========================
# Data Visualization
# =========================

# Set global style
sns.set_style("whitegrid")

# Sales by Region
region_sales.plot(kind='bar', color='skyblue')
plt.title("Sales by Region", fontsize=14)
plt.xlabel("Region")
plt.ylabel("Sales Amount")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("images/sales_by_region.png")
plt.show()

# Product Category Sales
sns.barplot(x=category_sales.index, y=category_sales.values,color='yellow')
plt.title("Sales by Product Category", fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("images/category_sales.png")
plt.show()

# Monthly Trend
monthly_sales.plot(kind='line', marker='o', color='green')
plt.title("Monthly Sales Trend", fontsize=14)
plt.grid(True)
plt.savefig("images/monthly_trend.png")
plt.show()

# Weekly Trend
weekly_sales.plot(kind='line', marker='o', color='orange')
plt.title("Weekly Sales Trend", fontsize=14)
plt.xlabel("Week")
plt.ylabel("Sales Amount")
plt.grid(True)
plt.savefig("images/weekly_sales.png")
plt.show()

# Monthly Trend (Year-Month)
monthly_sales_full.plot(kind='line', marker='o', color='purple')
plt.title("Monthly Sales Trend (Year-Month)", fontsize=14)
plt.grid(True)
plt.savefig("images/monthly_sales_full.png")
plt.show()

# Profit Distribution
sns.histplot(df['Profit'], kde=True, color='blue')
plt.title("Profit Distribution", fontsize=14)
plt.savefig("images/profit_distribution.png")
plt.show()

# Profit vs Loss
plt.bar(['Profit', 'Loss'], [total_profit, abs(total_loss)], color=['green', 'red'])
plt.title("Profit vs Loss", fontsize=14)
plt.savefig("images/profit_vs_loss.png")
plt.show()