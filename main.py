import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Function to calculate inflation_adjust price
def adjust_for_inflation(price, inflation_rate):
    return round(price * (1 + inflation_rate/100), 2)

# Read the CSV file
data = pd.read_csv('rice_beef_coffee_price_changes.csv')

# Filling values for 2006,2022
data.loc[data['Year'] == 2006, 'Inflation_rate'] = 31.96
data.loc[data['Year'] == 2022, 'Inflation_rate'] = 7.41

# Filling values for 2006,2022
data.loc[data['Year'].isin([2006, 2022]), 'Price_rice_infl'] = \
    data.loc[data['Year'].isin([2006, 2022])].apply(lambda row: adjust_for_inflation(row['Price_rice_kilo'], row['Inflation_rate']),axis=1)
data.loc[data['Year'].isin([2006, 2022]), 'Price_coffee_infl'] = \
    data.loc[data['Year'].isin([2006, 2022])].apply(lambda row: adjust_for_inflation(row['Price_coffee_kilo'], row['Inflation_rate']),axis=1)
data.loc[data['Year'].isin([2006, 2022]), 'Price_beef_infl'] = \
    data.loc[data['Year'].isin([2006, 2022])].apply(lambda row: adjust_for_inflation(row['Price_beef_kilo'], row['Inflation_rate']),axis=1)

# Transform to dateTime
data['Date'] = pd.to_datetime(data[['Year', 'Month']].assign(day=1))

# Set the datetime column as the index
data.set_index('Date', inplace=True)

# Drop the original nominal price columns and the inflation rate column
data.drop(columns=['Year', 'Month', 'Price_rice_kilo', 'Price_beef_kilo', 'Price_coffee_kilo', 'Inflation_rate'], inplace=True)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.index, data, test_size=0.2, shuffle=False)

print(data.info())

# Train a SVR model for each target variable on the training set
models = {}
for col in data.columns:
    models[col] = make_pipeline(StandardScaler(), SVR(kernel='linear', C=10, epsilon=0.1, gamma='scale'))
    models[col].fit(X_train.to_numpy().reshape(-1, 1), y_train[col])

# Generate a date range for the next 50 years
start_date = pd.to_datetime('2023-03-26')
end_date = start_date + pd.DateOffset(years=50)
date_range = pd.date_range(start_date, end_date, freq='MS')

# Predict the prices for each product for each month in the date range
price_beef_infl_pred = models['Price_beef_infl'].predict(date_range.to_numpy().reshape(-1, 1))
price_rice_infl_pred = models['Price_rice_infl'].predict(date_range.to_numpy().reshape(-1, 1))
price_coffee_infl_pred = models['Price_coffee_infl'].predict(date_range.to_numpy().reshape(-1, 1))

# Store the predicted prices in a data frame to write into a csv
pred_df = pd.DataFrame({
    'Date': date_range,
    'Price_beef_infl': price_beef_infl_pred,
    'Price_rice_infl': price_rice_infl_pred,
    'Price_coffee_infl': price_coffee_infl_pred
})

# Set the date column as the index
pred_df.set_index('Date', inplace=True)

# Store the predicted prices to a CSV file
#pred_df.to_csv('predicted_prices.csv', float_format='%.2f')
print(pred_df)
