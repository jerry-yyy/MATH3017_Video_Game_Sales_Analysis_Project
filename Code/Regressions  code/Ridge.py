import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler

#预处理
#Read csv file
vgsales = pd.read_csv("Video_Games_Sales_as_at_22_Dec_2016.csv")
# Delete empty values
vgsales.dropna(inplace=True)
# Convert year from float to integer
vgsales['Year_of_Release'] = vgsales['Year_of_Release'].astype(int)
# Convert user rating from string to float
vgsales['User_Score'] = vgsales['User_Score'].astype(float)
# Define sales area list
sales_columns = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']
# Initialize the dictionary that stores prediction results
results_summary = {}
#Create a StandardScaler object
scaler = StandardScaler()
# Define the columns to be standardized
numeric_columns = vgsales.columns[11:14]
# Standardize numeric variables
vgsales[numeric_columns] = scaler.fit_transform(vgsales[numeric_columns])
# Output standardized data
print(vgsales.head())
for sales_col in sales_columns:
    vgRidge = vgsales[
        ['Genre', 'Platform', 'Publisher', 'Year_of_Release', sales_col, 'Rating', 'Critic_Score', 'Critic_Count',
         'User_Score', 'User_Count', 'Developer']]
    # Take only the left tail of the sales data
    restrictionSales = (vgRidge[sales_col] < 2)
    vgRidge = vgRidge[restrictionSales]
    vgRidge[sales_col] = vgRidge[sales_col] ** (1 / 7)

    vgRidge = pd.concat([pd.get_dummies(vgRidge[['Genre', 'Platform', 'Publisher', 'Rating']]),
                         vgRidge[['Year_of_Release','Critic_Score','Critic_Count','User_Score','User_Count', sales_col]]], axis=1)

    data = vgRidge.values
    X, y = data[:, :-1], data[:, -1]

    # Define model
    model = Ridge(alpha=1.0)

    # Divide the data set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1234)

    # Fit model
    model.fit(X_train, y_train)

    # Define model evaluation method
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
    grid = dict()
    grid['alpha'] = np.logspace(-1, 1, num=50)
    search = GridSearchCV(model, grid, scoring='r2', cv=cv, n_jobs=-1)

    # Perform search
    results = search.fit(X, y)
    feature_coefs = model.coef_
    # Get the index of the largest 10 coefs
    top_10_indices = np.argsort(feature_coefs)[-10:][::-1]

    # Get the largest 10 coefs and corresponding feature names
    top_10_coefs = feature_coefs[top_10_indices]
    top_10_features = vgRidge.columns[top_10_indices]

    model = Ridge(alpha=results.best_params_['alpha'])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Update result dictionary
    results_summary[sales_col] = {

        'r2': results.best_score_,
        'Config': results.best_params_,
        'MSE': mean_squared_error(y_test, model.predict(X_test)),
        'Top 10 Coefs': top_10_coefs,
        'Top 10 Features': top_10_features
    }

# Output the summarized prediction results
for sales_col, metrics in results_summary.items():
    print(f"Results for {sales_col}:")
    print(f"r2: {metrics['r2']:.3f}")
    print(f"Config: {metrics['Config']}")
    print(f"MSE: {metrics['MSE']}")
    print("Top 10 Coefs:")
    for feature, coef in zip(metrics['Top 10 Features'], metrics['Top 10 Coefs']):
        print(f"{feature}: {coef}")
    all_coefs = model.coef_

    print("------------------------------------------------------")

import matplotlib.pyplot as plt


mse_values = []
r2_values = []
sales_regions = []

for sales_col, metrics in results_summary.items():
    sales_regions.append(sales_col)
    mse_values.append(metrics['MSE'])
    r2_values.append(metrics['r2'])


plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.bar(sales_regions, mse_values, color='blue')
plt.title('Mean Squared Error by Region')
plt.xlabel('Sales Region')
plt.ylabel('MSE')

plt.subplot(1, 2, 2)
plt.bar(sales_regions, r2_values, color='green')
plt.title('R^2 Score by Region')
plt.xlabel('Sales Region')
plt.ylabel('R^2 Score')

plt.tight_layout()
plt.show()


from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import numpy as np


alphas = np.logspace(-2, 2, 200)


coefs = []


for a in alphas:
    ridge = Ridge(alpha=a, fit_intercept=False)
    ridge.fit(X_train, y_train)
    coefs.append(ridge.coef_)


plt.figure(figsize=(10, 6))
ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
plt.show()






