import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('new_data_without_outliers.csv')

# Count the number of popular games on each platform
platform_counts = df['Platform'].value_counts()

# Find the platform with the highest number of popular games
most_popular_platform = platform_counts.idxmax()

# Create a table for platform counts
platform_counts_table = platform_counts.reset_index()
platform_counts_table.columns = ['Platform', 'Count']
print(platform_counts_table)

# Plot a bar chart
plt.figure(figsize=(10, 6))
plt.bar(platform_counts.index, platform_counts.values)
plt.xlabel('Platform')
plt.ylabel('Count')
plt.title('Number of Popular Games per Platform')
plt.xticks(rotation=45)
plt.show()