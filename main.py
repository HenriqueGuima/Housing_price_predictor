import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
import matplotlib.pyplot as plt

# Load and preprocess data
data = pd.read_csv('data/portugal_apartments.csv')
d_c = data.copy()

# Price column treatment
mask = d_c['Price'] != 'Preçosobconsulta'
d_c = d_c[mask]

# Drop index column
d_c.drop('Index', axis=1, inplace=True)

# Type column treatment
d_c.rename(columns={'Type': 'Rooms'}, inplace=True)
d_c['Rooms'] = d_c['Rooms'].str.replace('T', '')

# Area column treatment
d_c['Area'] = d_c['Area'].str.replace(' ', '.')

# Price column treatment
d_c['Price'] = d_c['Price'].str.replace(',', '')
d_c['Price'] = d_c['Price'].str.replace(' ', '')
d_c['Price'] = d_c['Price'].str.replace('.', '')
d_c['Price'] = pd.to_numeric(d_c['Price'])
d_c['Area'] = pd.to_numeric(d_c['Area'])

# Create a new column called price_per_sqm
d_c['Price_m2'] = d_c['Price'] / d_c['Area']
d_c['Price_m2'] = d_c['Price_m2'].round(2)

# One-hot encoding for location
d_c = pd.get_dummies(d_c, columns=['Location'])

# Define the features (X) and the target (y)
X = d_c[['Price_m2'] + [col for col in d_c.columns if col.startswith('Location_')]]
y = d_c['Price']  # Ensure `Price` is the target variable

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define and fit the pipeline
pipeline = Pipeline([
    ('std_scalar', StandardScaler()),  # Standardize features
    ('lin_reg', LinearRegression())  # Linear regression model
])

pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Extract the model and coefficients
lin_reg_model = pipeline.named_steps['lin_reg']
coeff_df = pd.DataFrame(lin_reg_model.coef_, X.columns, columns=['Coefficient'])

# Output the results
print("Intercept:", lin_reg_model.intercept_)
print(coeff_df)

# Evaluate the model
def print_evaluate(true, predicted):
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(mse)
    r2_square = metrics.r2_score(true, predicted)
    return mae, mse, rmse, r2_square

def cross_val(model):
    pred = cross_val_score(model, X, y, cv=10)
    return pred.mean()

# Plotting
plt.figure(figsize=(12, 6))

# Scatter plot of true vs. predicted values
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs. Predicted Values')

# Plotting the error distribution
plt.subplot(1, 2, 2)
errors = y_test - y_pred
pd.Series(errors).plot.kde()
plt.title('Error Distribution')
plt.xlabel('Error')

plt.tight_layout()
plt.show()

test_pred = pipeline.predict(X_test)
train_pred = pipeline.predict(X_train)

print('Test set evaluation:\n_____________________________________')
mae, mse, rmse, r2_square = print_evaluate(y_test, test_pred)
print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'R2 Square: {r2_square}')
print('_____________________________________')
print('Train set evaluation:\n_____________________________________')
mae, mse, rmse, r2_square = print_evaluate(y_train, train_pred)
print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'R2 Square: {r2_square}')

# Create the results DataFrame
results_df = pd.DataFrame(data=[["Linear Regression", *print_evaluate(y_test, test_pred) , cross_val(LinearRegression())]], 
                          columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])

print(results_df)
