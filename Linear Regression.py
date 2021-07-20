import pandas as pd #For data related tasks
import matplotlib.pyplot as plt #for data visualization 
import quandl #Stock market API for fetching Data
from sklearn.linear_model import LinearRegression
import numpy as np


quandl.ApiConfig.api_key = 'x5K8sX42s4MkEp1y1zmY' 
data = quandl.get('NSE/TCS')


from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10


#for normalizing data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

dataset=pd.DataFrame(data)
##print(dataset.head())
##Now we convert into csv
dataset.to_csv('TCS.csv')

dataset = pd.read_csv('TCS.csv')
#setting index as date
dataset['Date'] = pd.to_datetime(dataset.Date,format='%Y-%m-%d')
dataset.index = dataset['Date']

print(dataset.head())
print(data.isnull().sum())

##plt.figure(figsize=(16,8))
##plt.plot(dataset['Close'], label='Close Price history')
##plt.show()

#sorting
data = dataset.sort_index(ascending=True, axis=0)

#only column and date ke liye alag se bna rha huim data set
new_data = dataset.filter(['Date','Close'], axis=1)

for i in range(0,len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]

#create features


from fastai.tabular import add_datepart
add_datepart(new_data, 'Date')
new_data.drop('Elapsed', axis=1, inplace=True)  #elapsed will be the time stamp


new_data['mon_fri'] = 0
for i in range(0,len(new_data)):
    if (new_data['Dayofweek'][i] == 0 or new_data['Dayofweek'][i] == 4):
        new_data['mon_fri'][i] = 1
    else:
        new_data['mon_fri'][i] = 0


##This creates features such as:
##
##‘Year’, ‘Month’, ‘Week’, ‘Day’, ‘Dayofweek’, ‘Dayofyear’,
##‘Is_month_end’, ‘Is_month_start’,
##‘Is_quarter_end’, ‘Is_quarter_start’,  ‘Is_year_end’, and  ‘Is_year_start
##

print(new_data.head())


#split into train and validation
#split into train and validation
train = new_data[:987]
valid = new_data[987:]

x_train = train.drop('Close', axis=1)
y_train = train['Close']
x_valid = valid.drop('Close', axis=1)
y_valid = valid['Close']

#implement linear regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)

#make predictions and find the rmse
preds = model.predict(x_valid)

rms=np.sqrt(np.mean(np.power((np.array(y_valid)-np.array(preds)),2)))
print("rms: ",rms)

#plot
valid['Predictions'] = 0
valid['Predictions'] = preds

valid.index = new_data[987:].index
train.index = new_data[:987].index

plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.show()
