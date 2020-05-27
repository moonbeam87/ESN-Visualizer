import pandas as pd
from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from fbprophet import Prophet
import numpy as np
from tqdm import tqdm
import time
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from pyESN import ESN 
from matplotlib import rc


#Implementation based off: https://towardsdatascience.com/predicting-stock-prices-with-echo-state-networks-f910809d23d4

df = pd.read_csv('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=' + 'AMZN' +'&apikey=WCXVE7BAD668SJHL&datatype=csv&outputsize=full')
df = df.rename(columns={"timestamp":"Date"})
df = df.set_index(df['Date'])
df = df.sort_index()
df = df.drop(columns=['open', 'low', 'high', 'volume'])
df = df.drop(columns=['Date'])
df = df.reset_index()
df = df.drop(columns=['Date'])
df = df.reset_index()
df = df.rename(columns={"index":"x", "close":"y"})
print(df.head())
print(df.tail())

n_resevoir = 500
sparsity = 0.2
rand_seed = 23
spectral_radius = 1.2
noise = 0.0005

df = df[3431::]
df = df.reset_index()
df = df.drop(columns=['index', 'x'])
df = df.reset_index()
df = df.rename(columns={"index":"x"})

esn = ESN(n_inputs=1,n_outputs=1,n_reservoir=n_resevoir,sparsity=sparsity,random_state=rand_seed,spectral_radius=spectral_radius,noise=noise)
#First training will be with 1,500 datapoints to test accuracy
trainlen = 1500
#We want to predict the next day
future = 1
#we want to keep predicting this way for the next 100 points
futureTotal = 100


predictedTotal = np.zeros(futureTotal)
y = df['y']
y = y.to_numpy()
print('y data:')
print(type(y))
print(y)
#travers futureTotal by future days at a time

def predict(i, future, trainlen):
    predictedTraining = esn.fit(np.ones(trainlen), y[i:trainlen+i])
    prediction = esn.predict(np.ones(future))
    predictedTotal[i:i+future] = prediction[:,0]
start_time = time.time()

for i in range(trainlen):
    predict(i, future, trainlen)
print("--- %s seconds ---" % (time.time() - start_time))
rc('text', usetex=False)
plt.figure(figsize=(16,8))
plt.plot(range(0,trainlen+futureTotal),y[1000:trainlen+futureTotal],'b',label="Data", alpha=0.3)
#plt.plot(range(0,trainlen),pred_training,'.g',  alpha=0.3)
plt.plot(range(trainlen,trainlen+futureTotal),predictedTotal,'k',  alpha=0.8, label='Free Running ESN')

lo,hi = plt.ylim()
plt.plot([trainlen,trainlen],[lo+np.spacing(1),hi-np.spacing(1)],'k:', linewidth=4)

plt.title(r'Ground Truth and Echo State Network Output', fontsize=25)
plt.xlabel(r'Time (Days)', fontsize=20,labelpad=10)
plt.ylabel(r'Price ($)', fontsize=20,labelpad=10)
plt.legend(fontsize='xx-large', loc='best')
plt.savefig('/home/homeuser/Stonks/ESN.png')