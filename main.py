import pandas
import numpy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone
pandas.set_option('display.max_rows', None)
# from sklearn.discriminant_analysis import StandardScaler
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, ParameterGrid, RepeatedKFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Lasso

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

# Histogram 
d_c.hist(figsize=(12,12))
# plt.show()

X = d_c.drop(['Price_m2'], axis=1)
y = d_c['Price_m2']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

# Re-scale the filtered training data
X_train_scaled_filtered = scaler.fit_transform(X_train_filtered)
y_train_scaled_filtered = scaler.fit_transform(y_train_filtered.values.reshape(-1, 1)).flatten()

# Hyperparamenter Tuning to find the optimal alpha value for the Lasso regression model
alpha_tune = {
    'alpha': numpy.linspace(start=0.00001, stop=10, num=500),
    'tol': [1e-6, 1e-7, 1e-8]
    }

# TODO: Try another model to see if it performs better

# Progress: 1499/22500
# Progress: 1500/22500
# MSE: 90571725374414.02
# R2: -1602258.16

model_tuner = Lasso(fit_intercept=True, max_iter=1000000)
cross_validation = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)

class ProgressCallback:
    def __init__(self, total):
        self.total = total
        self.current = 0
        
    def __call__(self, params):
        self.current += 1
        print(f"Progress: {self.current}/{self.total}")

total_params = len(alpha_tune['alpha']) * len(alpha_tune['tol']) * cross_validation.get_n_splits()

def evaluate_param(param, X, y):
    model = clone(model_tuner)
    model.set_params(**param)
    model.fit(X, y)
    return model

params_grid = list(ParameterGrid(alpha_tune))
progress_callback = ProgressCallback(total_params)

results = Parallel(n_jobs=-1, backend='loky')(delayed(evaluate_param)(param, X_train_scaled_filtered, y_train_scaled_filtered) for param in params_grid)
for i, param in enumerate(params_grid):
    progress_callback(param)

# Find the best model based on scoring
best_model_index = numpy.argmax([mean_squared_error(y_train_scaled_filtered, res.predict(X_train_scaled_filtered)) for res in results])
best_model = results[best_model_index]
best_alpha = params_grid[best_model_index]

# Train best model
best_model.fit(X_train_scaled, y_train)

# Prediction
y_pred_scaled = best_model.predict(X_test_scaled)

# Inverse transform predictions and target to original scale
y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

pandas.DataFrame(X_train_scaled_filtered).to_csv('data/portugal_apartments_train.csv', index=False)
pandas.DataFrame(X_test_scaled).to_csv('data/portugal_apartments_test.csv', index=False)

# Model's Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.2f}')
print(f'R2 Score: {r2:.2f}')


