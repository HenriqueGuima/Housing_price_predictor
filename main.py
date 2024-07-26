import pandas
import numpy
import matplotlib.pyplot as plt
import seaborn as sns
pandas.set_option('display.max_rows', None)
# from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import GridSearchCV, RepeatedKFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

# Load the data
data = pandas.read_csv('data/portugal_apartments.csv')

# Copy the data
d_c = data.copy()

# Price column treatment
mask = d_c['Price'] != 'Preçosobconsulta'
d_c = d_c[mask]

# Drop index column
d_c.drop('Index', axis=1, inplace=True)

# Type column treatment
# Rename the column to Rooms
d_c.rename(columns={'Type': 'Rooms'}, inplace=True)
# Remove the 'T' from the column
d_c['Rooms'] = d_c['Rooms'].str.replace('T', '')

# Area column treatment
# print(d_c[d_c['Area'].str.contains(' ')])
d_c['Area'] = d_c['Area'].str.replace(' ', '.')

# Price and Area to numeric values
d_c['Price'] = d_c['Price'].str.replace(',', '')
d_c['Price'] = d_c['Price'].str.replace(' ', '')
d_c['Price'] = d_c['Price'].str.replace('.', '')
d_c['Price'] = pandas.to_numeric(d_c['Price'])
d_c['Area'] = pandas.to_numeric(d_c['Area'])

# Create a new column called price_per_sqm
d_c['Price_m2'] = d_c['Price'] / d_c['Area']
d_c['Price_m2'] = d_c['Price_m2'].round(2)
d_c.to_csv('data/portugal_apartments_cleaned.csv')

# Hot encoding
d_c = pandas.get_dummies(d_c, columns=['Location'])

# # Data Visualization
# # Mean price per Location
# mean_price_location = d_c.groupby('Location')['Price_m2'].mean().sort_values(ascending=False)
# mean_price_location.plot(kind='bar')
# # plt.show()

# # Mean price per number of rooms
# mean_price_rooms = d_c.groupby('Rooms')['Price_m2'].mean().sort_values(ascending=False)
# mean_price_rooms.plot(kind='bar')
# # plt.show()

# Histogram 
d_c.hist(figsize=(12,12))
# plt.show()

# # Convert dsc['Price_m2'] to numerical values
# d_c['Price_m2'] = pandas.to_numeric(d_c['Price_m2'], errors='coerce')

# # Change data type of the column price_m2 to float
# d_c['Price_m2'] = d_c['Price_m2'].astype(int)

# # Shuffle rows
# dsc = d_c.sample(frac = 1)

X = d_c.drop(['Price_m2'], axis=1)
y = d_c['Price_m2']

# Split the data
# df_train, df_test = train_test_split(dsc, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Create a categorical version for the price_m2 variable
# dsc['Price_m2_cat'] = pandas.cut(dsc['Price_m2'], bins=5, labels=False)

# Random splitting without stratification
# df_train, df_test = train_test_split(dsc, test_size=0.2, random_state=42)

# check the proportions of the data
# df_train_tc = df_train['Price_m2'].value_counts() / len(df_train)
# df_test_tc = df_test['Price_m2'].value_counts() / len(df_test)

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Remove outliers
def remove_outliers(df, col):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

# Remove outliers from y_train
y_train_filtered = remove_outliers(pandas.DataFrame(y_train, columns=['Price_m2']), 'Price_m2')
X_train_filtered = X_train.loc[y_train_filtered.index]

# y_train_scaled = scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
# y_test_scaled = scaler.transform(y_test.values.reshape(-1, 1)).flatten()

# Re-scale the filtered training data
X_train_scaled_filtered = scaler.fit_transform(X_train_filtered)
y_train_scaled_filtered = scaler.fit_transform(y_train_filtered.values.reshape(-1, 1)).flatten()

# Hyperparamenter Tuning to find the optimal alpha value for the Lasso regression model
alpha_tune = {
    'alpha': numpy.linspace(start=0.00001, stop=1, num=300),
    'tol': [1e-6, 1e-7, 1e-8]
    }

model_tuner = Lasso(fit_intercept=True, max_iter=1000000)
cross_validation = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)

grid_search = GridSearchCV(model_tuner, alpha_tune, cv=cross_validation, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

best_model = grid_search.best_estimator_
best_alpha = grid_search.best_params_

# Train best model
best_model.fit(X_train_scaled, y_train)

# Prediction
y_pred_scaled = best_model.predict(X_test_scaled)

# Inverse transform predictions and target to original scale
y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_test = scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()

# print('========== BEFORE ===========')
# print(df_train_tc)
# print(df_test_tc)
# print('=============================')

# Save the data
# X_train_scaled.to_csv('data/portugal_apartments_train.csv')
# X_test_scaled.to_csv('data/portugal_apartments_test.csv')

pandas.DataFrame(X_train_scaled).to_csv('data/portugal_apartments_train.csv', index=False)
pandas.DataFrame(X_test_scaled).to_csv('data/portugal_apartments_test.csv', index=False)

# Linear Regression
# lr_model = LinearRegression()
# lr_model.fit(df_train[['Area']], df_train['Price_m2'])

# lasso_model = Lasso(alpha=0.1)  # alpha is the regularization strength
# lasso_model.fit(df_train[['Area']], df_train['Price_m2'])

# y_pred_lasso = lasso_model.predict(df_test[['Area']])

# y_pred = lr_model.predict(df_test[['Area']])
# y_test = df_test['Price_m2']

### BEFORE REGULARIZATION ###
# MSE: 2206294586.78
# R2: -0.00

### AFTER REGULARIZATION ###
# MSE: 7855009.63
# R2: -0.43

# Model's Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MSE: {mse:.2f}')
print(f'R2: {r2:.2f}')


