import pandas as pd

# Read the data file
data = pd.read_csv('new_data_without_outliers.csv')

# Analyze Japanese market sales
publisher_JP = pd.DataFrame(data['JP_Sales'].groupby(data['Publisher']).sum()).sort_values(by='JP_Sales', ascending=False)

# Output the sales situation in the Japanese market
print("Sales in the Japanese market:")
print(publisher_JP)