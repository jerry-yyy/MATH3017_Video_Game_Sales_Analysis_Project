import pandas as pd

data = pd.read_csv('new_data_without_outliers.csv')

market_sales = data.groupby('Genre')['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'].sum()

print(market_sales)