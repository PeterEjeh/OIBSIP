import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset to inspect its structure
file_path = 'Warehouse_and_Retail_Sales.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset to understand its structure
data.head(), data.info(), data.describe()
print(data.info())

# Data Cleaning

# Handle missing values: Drop rows with missing values as they are minimal.
data_cleaned = data.dropna()

# Ensure all numeric values are non-negative for meaningful analysis
columns_to_check = ["RETAIL SALES", "RETAIL TRANSFERS", "WAREHOUSE SALES"]
for column in columns_to_check:
    data_cleaned = data_cleaned[data_cleaned[column] >= 0]

# Descriptive Statistics
descriptive_stats = data_cleaned[columns_to_check].describe()

# Perform time-based aggregation to analyze trends
data_cleaned['DATE'] = pd.to_datetime(data_cleaned[['YEAR', 'MONTH']].assign(DAY=1))
monthly_sales = data_cleaned.groupby('DATE')[['RETAIL SALES', 'WAREHOUSE SALES']].sum()

descriptive_stats, monthly_sales.head()
print(descriptive_stats)
print(monthly_sales)

# Visualization: Sales Trends Over Time with Custom Colors
plt.figure(figsize=(14, 7))

# Custom colors for lines
colors = ['#1f77b4', '#ff7f0e']  # Blue for Retail, Orange for Warehouse

# Plot each series with its own color
for i, column in enumerate(monthly_sales.columns):
    sns.lineplot(data=monthly_sales[column], markers=True, color=colors[i], label=column)

plt.title('Monthly Sales Trends (Retail and Warehouse)', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Sales ($)', fontsize=12)
plt.legend(fontsize=12)
plt.grid(visible=True)
plt.tight_layout()
plt.show()

# Analyze Customer and Product Data
item_sales = data_cleaned.groupby('ITEM TYPE')[['RETAIL SALES', 'WAREHOUSE SALES']].sum()
top_suppliers = data_cleaned.groupby('SUPPLIER')[['RETAIL SALES', 'WAREHOUSE SALES']].sum().nlargest(10, 'RETAIL SALES')

# Individual Visualization: Retail Sales by Item Type
plt.figure(figsize=(14, 7))
sns.barplot(x=item_sales.index, y='RETAIL SALES', data=item_sales.reset_index())
plt.title('Retail Sales by Item Type', fontsize=16)
plt.xlabel('Item Type', fontsize=12)
plt.ylabel('Retail Sales ($)', fontsize=12)
plt.xticks(rotation=45)  # Rotate x-labels for better readability if many categories
plt.tight_layout()
plt.show()

# Individual Visualization: Top 10 Suppliers by Retail Sales
plt.figure(figsize=(14, 7))
sns.barplot(y=top_suppliers.index, x='RETAIL SALES', data=top_suppliers.reset_index())
plt.title('Top 10 Suppliers by Retail Sales', fontsize=16)
plt.xlabel('Retail Sales ($)', fontsize=12)
plt.ylabel('Supplier', fontsize=12)
plt.tight_layout()
plt.show()

# Prepare Data for Heatmap
sales_by_month_year = data_cleaned.pivot_table(
    index='YEAR', columns='MONTH', values='RETAIL SALES', aggfunc='sum'
)

# Heatmap Visualization
plt.figure(figsize=(12, 6))
sns.heatmap(sales_by_month_year, cmap="YlGnBu", annot=True, fmt=".0f")
plt.title('Heatmap of Retail Sales by Year and Month', fontsize=16)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Year', fontsize=12)
plt.show()

# Recommendations based on insights
recommendations = """
        RECOMMENDATIONS
        
1. **Focus on High-Performing Product Categories**:
   - Increase marketing and inventory for items like WINE and BEER, which are top contributors to retail sales.

2. **Strengthen Relationships with Top Suppliers**:
   - Collaborate with the top 10 suppliers to negotiate better terms or exclusive deals to sustain high sales levels.

3. **Seasonal Promotions**:
   - Use the seasonal sales patterns identified in the time series and heatmap analysis to plan promotions during peak periods.

4. **Inventory Optimization**:
   - Leverage sales data to ensure sufficient stock of high-demand products during peak months.

5. **Expand Supplier Base**:
   - Explore opportunities with underperforming suppliers or add new suppliers to diversify product offerings.
"""

print(recommendations)