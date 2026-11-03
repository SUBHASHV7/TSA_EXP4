# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES

## Name: SUBHASH V
## Reg.No: 212224240163
## Date: 3.11.2026

### AIM:
To implement ARMA model in python.

### SOFTWARE REQUIRED: 
1. Google Colab/
2. Jupiter Notebook

### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000 data points using the ArmaProcess class. Plot the generated time series and set the title and x-axis limits.
4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000 data points using the ArmaProcess class. Plot the generated time series and set the title and x-axis limits.
6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using plot_acf and plot_pacf.

### PROGRAM:
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load dataset
data = pd.read_csv('/content/Indian_Climate_Dataset_2024_2025.csv')
data['Date'] = pd.to_datetime(data['Date'])

# Pick one city and sort by date (change the city if you want)
city = data['City'].unique()[0]
data = data[data['City'] == city].sort_values('Date').reset_index(drop=True)

# Declare required variables and set figure size, and visualise the data
N = 1000
plt.rcParams['figure.figsize'] = [12, 6]  # plt.rcParams is a dictionary-like object in Matplotlib
X = data['Temperature_Avg (°C)']
plt.plot(X)
plt.title(f'Original Data - {city}')
plt.show()

plt.subplot(2, 1, 1)
plot_acf(X, lags=len(X)//4, ax=plt.gca())
plt.title('Original Data ACF')

plt.subplot(2, 1, 2)
plot_pacf(X, lags=len(X)//4, ax=plt.gca())
plt.title('Original Data PACF')
plt.tight_layout()
plt.show()

# Fitting the ARMA(1,1) model and deriving parameters
arma11_model = ARIMA(X, order=(1, 0, 1)).fit()
phi1_arma11 = arma11_model.params['ar.L1']
theta1_arma11 = arma11_model.params['ma.L1']

# Simulate ARMA(1,1) Process
ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1, theta1_arma11])
ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)
plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, 500])
plt.show()

# Plot ACF and PACF for ARMA(1,1)
plot_acf(ARMA_1)
plt.show()
plot_pacf(ARMA_1)
plt.show()

# Fitting the ARMA(2,2) model and deriving parameters
arma22_model = ARIMA(X, order=(2, 0, 2)).fit()
phi1_arma22 = arma22_model.params['ar.L1']
phi2_arma22 = arma22_model.params['ar.L2']
theta1_arma22 = arma22_model.params['ma.L1']
theta2_arma22 = arma22_model.params['ma.L2']

# Simulate ARMA(2,2) Process
ar2 = np.array([1, -phi1_arma22, -phi2_arma22])
ma2 = np.array([1, theta1_arma22, theta2_arma22])
ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N*10)
plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')
plt.xlim([0, 500])
plt.show()

# Plot ACF and PACF for ARMA(2,2)
plot_acf(ARMA_2)
plt.show()
plot_pacf(ARMA_2)
plt.show()
```

### OUTPUT:

SIMULATED ARMA(1,1) PROCESS:

<img width="993" height="528" alt="download" src="https://github.com/user-attachments/assets/a1616ff5-5f50-4c07-a484-8f28a1b0a459" />


Partial Autocorrelation

<img width="1002" height="528" alt="download" src="https://github.com/user-attachments/assets/4aa6e6f9-cc1e-4b0e-99ce-15e058894ee4" />


Autocorrelation

<img width="1002" height="528" alt="download" src="https://github.com/user-attachments/assets/0c7cec3e-d40d-4664-aabf-46f0cb061963" />


SIMULATED ARMA(2,2) PROCESS:

<img width="993" height="528" alt="download" src="https://github.com/user-attachments/assets/ed418153-61c5-4fa4-9f41-afd02a4018b4" />


Partial Autocorrelation

<img width="1002" height="528" alt="download" src="https://github.com/user-attachments/assets/7a0a0f00-c609-497d-832f-b0144521c2c0" />


Autocorrelation

<img width="1002" height="528" alt="download" src="https://github.com/user-attachments/assets/5884305a-bf4e-496a-9780-53ff712df7b9" />


### RESULT:
Thus, a python program is created to fir ARMA Model successfully.
