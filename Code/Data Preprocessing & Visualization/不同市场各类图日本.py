import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('new_data_without_outliers.csv')

na_sales = data.groupby('Genre')['JP_Sales'].sum()

na_sales_sorted = na_sales.sort_values()

fig, ax = plt.subplots()

na_sales_sorted.plot(kind='barh', ax=ax)

ax.set_title('Japanese Game Sales by Genre')
ax.set_xlabel('Sales (in millions)')
ax.set_ylabel('Genre')

plt.show()