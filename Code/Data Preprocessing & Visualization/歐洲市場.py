import pandas as pd

# Read the data file
data = pd.read_csv('new_data_without_outliers.csv')

# Analyze European market sales
publisher_EU = pd.DataFrame(data['EU_Sales'].groupby(data['Publisher']).sum()).sort_values(by='EU_Sales', ascending=False)

# Output the sales situation in the European market
print("Sales in the European market:")
print(publisher_EU)