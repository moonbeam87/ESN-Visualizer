#Based of the credits below:
"""
A minimalistic Echo State Networks demo with Mackey-Glass (delay 17) data 
in "plain" scientific Python.
by Mantas LukoÅ¡eviÄius 2012-2018
http://mantas.info
"""
import pandas as pd
from pandas_datareader import data
import datetime as dt
from tqdm import tqdm
import time
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from numpy import *
import matplotlib.pyplot as plt
import scipy.linalg

trainLen = 2000 #Use 2,000 points to train
testLen = 3000 
initLen = 100




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
y = df['y']
y = y.to_numpy()
print(y)

inSize = outSize = 1
resSize = 100
a = 0.3
random.seed(42)
Win = (random.rand(resSize,1+inSize)-0.5)*1
W = random.rand(resSize, resSize)-0.5

print('spectral radius')
rhoW = max(abs(linalg.eig(W)[0]))
print('done')
W = W * 1.25/rhoW


# allocated memory for the design (collected states) matrix
X = zeros((1+inSize+resSize,trainLen-initLen))
# set the corresponding target matrix directly
Yt = y[None,initLen+1:trainLen+1] 

# run the reservoir with the data and collect X
x = zeros((resSize,1))
for t in range(trainLen):
    u = y[t]
    x = (1-a)*x + a*tanh( dot( Win, vstack((1,u)) ) + dot( W, x ) )
    if t >= initLen:
        X[:,t-initLen] = vstack((1,u,x))[:,0]
    
# train the output by ridge regression
reg = 1e-8  # regularization coefficient
X_T = X.T
Wout = dot( dot(Yt,X_T), linalg.inv( dot(X,X_T) + \
    reg*eye(1+inSize+resSize) ) )

Y = zeros((outSize,testLen))
u = y[trainLen]
for t in range(testLen):
    x = (1-a)*x + a*tanh( dot( Win, vstack((1,u)) ) + dot( W, x ) )
    z = dot( Wout, vstack((1,u,x)) )
    Y[:,t] = z
    ## generative mode:
    #u = y
    ## predictive mode:
    u = y[trainLen+t+1] 

errorLen = 500
mse = sum( square( y[trainLen+1:trainLen+errorLen+1] - 
    Y[0,0:errorLen] ) ) / errorLen
print('MSE = ' + str( mse ))

def createList(r1, r2): 
  
    # Testing if range r1 and r2  
    # are equal 
    if (r1 == r2): 
        return r1 
  
    else: 
  
        # Create empty list 
        res = [] 
  
        # loop to append successors to  
        # list until r2 is reached. 
        while(r1 < r2+1 ): 
              
            res.append(r1) 
            r1 += 1
        return res 

plt.figure(figsize=(16,8))
plt.plot( y[trainLen::], 'g' )
print(Y.T)
Xoffset = createList(2000,4000)
plt.plot(Y.T, 'b' )
plt.title('Target and generated signals $y(n)$ starting at $n=0$')
plt.legend(['Target signal', 'Free-running predicted signal'])
plt.savefig('/home/dev/Stonks/ESN.png')