import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('new_data_without_outliers.csv')

genre_counts = data['Genre'].value_counts()

genre_shuffled = genre_counts.sample(frac=1)

fig, ax = plt.subplots()

ax.pie(genre_shuffled, labels=genre_shuffled.index, autopct='%1.1f%%', startangle=90, counterclock=False, wedgeprops={'linewidth': 1, 'edgecolor': 'white'})

ax.set_title('Game Genre Distribution')

ax.axis('equal')

plt.show()