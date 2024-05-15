import pandas
import numpy
import matplotlib.pyplot as plt
pandas.set_option('display.max_rows', None)

# Load the data
data = pandas.read_csv('C:/Users/lmars/Desktop/AI projects/Housing_price_predictor/data/portugal_apartments.csv')

#copy the data
d_c = data.copy()

# save as a new csv file
d_c.to_csv('C:/Users/lmars/Desktop/AI projects/Housing_price_predictor/data/portugal_apartments_cleaned.csv')

# Price column treatment
mask = d_c['Price'] != 'Preçosobconsulta'
d_c = d_c[mask]

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


