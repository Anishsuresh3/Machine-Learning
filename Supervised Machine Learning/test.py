import numpy as np
import pandas as pd
def sigmoid(z):
    g = 1/(1+ np.exp(-z))
    return g
w=[-0.02715896 , 0.08194569] 
b=-0.014439518556218094
df = pd.read_csv('./Datasets/habermann.csv')
df.columns=['age','year','axillary','survival']
df['survival'] = df['survival']-1
fd=pd.DataFrame(df,columns=['age','axillary'])
x_train = []
for i in range(241):
    x_train.append(fd.loc[i].tolist())
f_wb = sigmoid(np.dot(x_train,w)+b)
print(f_wb)
print(df['survival'].tolist()[:241])
