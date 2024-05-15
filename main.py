import pandas
import numpy
import matplotlib.pyplot as plt
import seaborn as sns
pandas.set_option('display.max_rows', None)
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

# Load the data
data = pandas.read_csv('C:/Users/lmars/Desktop/AI projects/Housing_price_predictor/data/portugal_apartments.csv')

#copy the data
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
d_c.to_csv('C:/Users/lmars/Desktop/AI projects/Housing_price_predictor/data/portugal_apartments_cleaned.csv')

# Data Visualization
# Mean price per Location
mean_price_location = d_c.groupby('Location')['Price_m2'].mean().sort_values(ascending=False)
mean_price_location.plot(kind='bar')
# plt.show()

# Mean price per number of rooms
mean_price_rooms = d_c.groupby('Rooms')['Price_m2'].mean().sort_values(ascending=False)
mean_price_rooms.plot(kind='bar')
# plt.show()

# Histogram 
d_c.hist(figsize=(12,12))
# plt.show()

# Convert dsc['Price_m2'] to numerical values
d_c['Price_m2'] = pandas.to_numeric(d_c['Price_m2'], errors='coerce')

# Change data type of the column price_m2 to float
d_c['Price_m2'] = d_c['Price_m2'].astype(int)

# Shuffle rows
dsc = d_c.sample(frac = 1)

# numerical_columns = dsc.select_dtypes(include=['number']).columns

# Split the data
dsc['Price_m2'] = dsc.sum(axis=1).round(2)
df_train, df_test = train_test_split(dsc, test_size=0.2, random_state=42)

dsc['Price_m2'] = pandas.cut(dsc['Price_m2'], bins=5, labels=False)
df_train, df_test = train_test_split(dsc, test_size=0.2, random_state=42)

# check the proportions of the data
df_train_tc = df_train['Price_m2'].value_counts() / len(df_train)
df_test_tc = df_test['Price_m2'].value_counts() / len(df_test)

print('========== BEFORE ===========')
print(df_train_tc)
print(df_test_tc)
print('=============================')

# Save the data
train.to_csv('C:/Users/lmars/Desktop/AI projects/Housing_price_predictor/data/portugal_apartments_train.csv')
test.to_csv('C:/Users/lmars/Desktop/AI projects/Housing_price_predictor/data/portugal_apartments_test.csv')

