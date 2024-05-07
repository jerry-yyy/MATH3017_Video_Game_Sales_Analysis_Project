import pandas as pd

# Read the data file
data = pd.read_csv('new_data_without_outliers.csv')

# Analyze North American market sales
publisher_NA = pd.DataFrame(data['NA_Sales'].groupby(data['Publisher']).sum()).sort_values(by='NA_Sales', ascending=False)

# Output the sales situation in the North American market
print("Sales in the North American market:")
print(publisher_NA)
