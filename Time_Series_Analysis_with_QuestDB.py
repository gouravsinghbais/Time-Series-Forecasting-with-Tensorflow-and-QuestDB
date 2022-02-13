## import dependencies
import numpy as np
import pandas as pd

## deep learning dependencies 
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *

## QuestDB dependencies
import io
import requests
import urllib.parse as par

## timestamp dependencies 
from datetime import datetime

## visualisation dependencies
import matplotlib.pyplot as plt
%matplotlib inline

# read dataset using pandas library
df = pd.read_excel('Excelrates.xlsx')
## check first few rows of the dataset
df.head()

## create table query 
q = 'create table excel_rates'\
    '(Date timestamp,'\
    'USD int,'\
    'INR double)'
## connect to QuestDB URL and execute the query
r = requests.get("http://localhost:9000/exec?query=" + q)

## print the status code once executed the table creation query
print(r.status_code)

## variables for tracking successful execution of queries 
success = 0
fail = 0

## iterate over each row and store it in the QuestDB table 
for i, row in df.iterrows():
    date = row['Date']
    ## convert date to datetime format to store in DB
    date = "'"+date.strftime('%Y-%m-%dT%H:%M:%S.%fZ')+"'"
    usd = row['USD']
    inr = row['INR']
    query = "insert into excel_rates values(" + date + "," + str(usd) + ","  + str(inr) +")"
    r = requests.get("http://localhost:9000/exec?query=" + query)
    if r.status_code == 200:
        success += 1
    else:
        fail += 1
        
## check if the execution is successful or not
if fail > 0:
    print("Rows Failed: " + str(fail))
if success > 0:
    print("Rows inserted: " + str(success))
    
    
## select data from QuestDB
r = requests.get("http://localhost:9000/exp?query=select * from excel_rates")
rawData = r.text

## convert Bytes to CSV format and read using pandas library
df = pd.read_csv(io.StringIO(rawData), parse_dates=['Date'])
df.columns 

## drop USD column from the dataframe
df = df.drop('USD', axis=1)
## convert Date column to datetime format
df['Date'] = pd.to_datetime(df["Date"])
## set Date as index 
indexed_df = df.set_index(["Date"], drop=True)
indexed_df.head()

## plot dataframe
indexed_df.plot()

## remove 0 values 
indexed_df = indexed_df[indexed_df.INR != 0.0]
indexed_df = indexed_df.iloc[::-1]
indexed_df.head()

## shift INR values by 1 
shifted_df= indexed_df.shift()
## merge INR and Shifter INR values 
concat_df = [indexed_df, shifted_df]
data = pd.concat(concat_df,axis=1)
## Replace NaNs with 0
data.fillna(0, inplace=True)
data.head()

## convert data to numpy array 
data = np.array(data)
## you can take last 500 data points as test set
train , test = data[0:-500], data[-500:]

# Scale
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)

# train data
y_train = train_scaled[:,-1]
X_train = train_scaled[:,0:-1]
X_train = X_train.reshape(len(X_train),1,1)

#test data
y_test = test_scaled[:,-1]
X_test = test_scaled[:,0:-1]

## GRU Model
model = Sequential()
## GRU layer 
model.add(GRU(75, input_shape=(1,1)))
## output layer 
model.add(Dense(1))
optimizer = Adam(lr=1e-3)
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, batch_size=20, shuffle=False)

## make predictions for test set 
X_test = X_test.reshape(500,1,1)
y_pred = model.predict(X_test)

## visualise results
plt.plot(y_pred, label = 'predictions')
plt.plot(y_test, label = 'actual')
plt.legend()

## visualize results
plt.plot(y_pred[:100], label = 'predictions')
plt.plot(y_test[:100], label = 'actual')
plt.legend()