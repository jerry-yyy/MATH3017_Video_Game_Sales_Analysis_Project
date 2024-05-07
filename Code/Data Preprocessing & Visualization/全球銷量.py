import pandas as pd

# Read the CSV file and create a DataFrame
df = pd.read_csv('new_data_without_outliers.csv')

# Group by publisher and calculate total sales
sales_by_publisher = df.groupby('Publisher').sum()['Global_Sales']

# Sort by total sales in descending order
sorted_sales = sales_by_publisher.sort_values(ascending=False)

# Get the top 10 publishers
top_10_publishers = sorted_sales.head(10)

# Create a table
table = pd.DataFrame({'Publisher': top_10_publishers.index, 'Global Sales (in millions)': top_10_publishers.values})

# Print the table
print(table)