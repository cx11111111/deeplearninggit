from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

data= pd.read_csv("C:\\Users\\22279\\Desktop\\数据集\\风机功率数据\\turb_23.csv")

base_date = datetime(2023, 1, 1)

# Adjust 'Day' to reflect the correct date by adding it as an offset to the base date
data['Corrected_Date'] = data['Day'].apply(lambda x: base_date + pd.Timedelta(days=x-1))

# Combine 'Corrected_Date' and 'Tmstamp' to create a complete datetime object
data['Complete_Datetime'] = data.apply(lambda row: datetime(row['Corrected_Date'].year, row['Corrected_Date'].month, row['Corrected_Date'].day, int(row['Tmstamp'].split(':')[0]), int(row['Tmstamp'].split(':')[1])), axis=1)

# Set this new datetime as the index
data.set_index('Complete_Datetime', inplace=True)


# Focusing on the 'Patv' column for ARIMA model
data = data[data['Patv'] >= 0]
print(data)
timeseries = data['Patv'].dropna() # Dropping NaN values
timeseries=timeseries.diff().dropna()
print(timeseries)

adf_test_result=adfuller(timeseries)
adf_p_value=adf_test_result[1]
print('p=',adf_p_value)

# Plot the Autocorrelation Function (ACF)
plt.figure()
plt.subplot(211)
plot_acf(timeseries, ax=plt.gca(),lags=144)  # You can adjust the number of lags as needed
plt.title('Autocorrelation Function')

# Plot the Partial Autocorrelation Function (PACF)
plt.subplot(212)
plot_pacf(timeseries, ax=plt.gca(),lags=144)  # Again, adjust the number of lags as needed
plt.title('Partial Autocorrelation Function')

plt.tight_layout()
plt.show()

# Splitting the data into training, validation, and test sets
total_size = len(timeseries)
train_size = int(total_size * 0.7)
validation_size = int(total_size * 0.15)
train_set = timeseries[:train_size]
validation_set = timeseries[train_size:train_size+validation_size]
test = timeseries[train_size+validation_size:]

# Building the ARIMA model
arima_model = ARIMA(train_set, order=(3,1,5))
arima_fit = arima_model.fit()


# Making predictions on the test set
prediction = arima_fit.forecast(steps=len(test))

prediction.index = test.index
print(prediction)

# Plotting the forecast against the actual values
plt.figure(figsize=(12,6))
plt.plot(test, label='Actual')
plt.plot(prediction,  label='Forecast')
plt.legend()
plt.title('ARIMA Model Rolling Forecast')
plt.show()

# # Creating a DataFrame for comparison
# test_comparison = pd.DataFrame({'Actual': test_set, 'Predicted': test_predictions})
#
# # Saving the predictions to a CSV file
# output_file_path = '/mnt/data/test_predictions.csv'
# test_comparison.to_csv(output_file_path)