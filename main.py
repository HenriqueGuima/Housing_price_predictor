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
X = d_c[['Rooms', 'Area']]
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
    # Adding Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((true - predicted) / true)) * 100
    return mae, mse, rmse, r2_square, mape

def cross_val(model):
    # Cross-validation with scoring as negative mean squared error
    pred = cross_val_score(model, X, y, cv=10, scoring='neg_mean_squared_error')
    return -pred.mean()

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

# Evaluate on both test and training sets
test_pred = pipeline.predict(X_test)
train_pred = pipeline.predict(X_train)

# Obtain metrics for training and test data
mae_test, mse_test, rmse_test, r2_test, mape_test = print_evaluate(y_test, test_pred)
mae_train, mse_train, rmse_train, r2_train, mape_train = print_evaluate(y_train, train_pred)

# Create a DataFrame for easy plotting
metrics_df = pd.DataFrame({
    'Set': ['Train', 'Test'],
    'MSE': [mse_train, mse_test],
    'RMSE': [rmse_train, rmse_test],
    'R2 Square': [r2_train, r2_test]
})

# Plot MSE and RMSE
fig, ax1 = plt.subplots()

# Create twin axes for plotting different scales
ax2 = ax1.twinx()

# Plot MSE
metrics_df.plot(kind='bar', x='Set', y='MSE', ax=ax1, color='lightblue', position=1, width=0.4, legend=False)
ax1.set_ylabel('Mean Squared Error (MSE)', color='lightblue')

# Plot RMSE
metrics_df.plot(kind='bar', x='Set', y='RMSE', ax=ax2, color='salmon', position=0, width=0.4, legend=False)
ax2.set_ylabel('Root Mean Squared Error (RMSE)', color='salmon')

# Add titles and labels
ax1.set_title('Model Evaluation Metrics')
ax1.set_xlabel('Data Set')
ax1.set_xticklabels(metrics_df['Set'], rotation=0)

# Add legend
ax1.legend(['MSE'], loc='upper left')
ax2.legend(['RMSE'], loc='upper right')

plt.show()

# Plot R2 Square
plt.figure(figsize=(8, 5))
metrics_df.plot(kind='bar', x='Set', y='R2 Square', color='lightgreen', legend=False)
plt.title('R2 Square Metric')
plt.xlabel('Data Set')
plt.ylabel('R2 Square')
plt.xticks(rotation=0)
plt.ylim(-1, 1)
# plt.show()