import pandas as pd

# Read CSV file
data = pd.read_csv('new_data_without_outliers.csv')

# Count the number of games of various types
genre_counts = data['Genre'].value_counts()

# Print results
print(genre_counts)