import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

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
d_c['Rooms'] = pd.to_numeric(d_c['Rooms'])

# Area column treatment
d_c['Area'] = d_c['Area'].str.replace(' ', '.')

# Price column treatment
d_c['Price'] = d_c['Price'].str.replace(',', '')
d_c['Price'] = d_c['Price'].str.replace(' ', '')
d_c['Price'] = d_c['Price'].str.replace('.', '')
d_c['Price'] = pd.to_numeric(d_c['Price'])
d_c['Area'] = pd.to_numeric(d_c['Area'])

apartments = d_c.copy()
numerical_data = apartments[['Rooms', 'Price', 'Area']]

X = apartments.iloc[:, :].values
Y = apartments.iloc[:, 1].values

# print(X)

#Correlation Matrix
sns.heatmap(numerical_data.corr(), annot=True, cmap='coolwarm')
plt.show()

#Label encoding
labelencoder = LabelEncoder()

X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categories = [3])

onehotencoder = ColumnTransformer(
    transformers=[('onehot', OneHotEncoder(), [3])],
    remainder='passthrough'
)

X = onehotencoder.fit_transform(X)
# X = np.array(X)

# Data Scaling
scaler = StandardScaler(with_mean=False)
X = scaler.fit_transform(X)

# print(X)

# Data split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Fitting the model
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting the test set results
Y_pred = regressor.predict(X_test)
print(Y_pred)

# Fitting the model with Lasso regression
# from sklearn.linear_model import Lasso
# regressor = Lasso(alpha=0.1)
# regressor.fit(X_train, Y_train)

# # Predicting the test set results
# Y_pred = regressor.predict(X_test)
# print(Y_pred)

# # Fitting the model with Ridge regression
# from sklearn.linear_model import Ridge
# regressor = Ridge(alpha=0.1)
# regressor.fit(X_train, Y_train)

# # Predicting the test set results
# Y_pred = regressor.predict(X_test)
# print(Y_pred)

# Calculating the coefficients and intercept
print('Coefficients: \n', regressor.coef_)
print('Intercept: \n', regressor.intercept_)

# Model evaluation
from sklearn.metrics import r2_score
r2 = r2_score(Y_test, Y_pred)
print('R2 Score: ', r2)

# Cross validation
cv_score = cross_val_score(regressor, X, Y, cv=5)
print('Cross Validation Score: ', cv_score)

# Visualizing the results
plt.scatter(Y_test, Y_pred, color='red')
plt.title('Predicted vs Actual')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()

print(apartments.info())

