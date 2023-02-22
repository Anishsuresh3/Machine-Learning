import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd

df = pd.read_csv('./Datasets/habermann.csv')
df.columns=['age','year','axillary','survival']
df['survival'] = df['survival']-1
fd=pd.DataFrame(df,columns=['age','year','axillary'])
x_train = []
for i in range(241):
    x_train.append(fd.loc[i].tolist())
y_train = df['survival'].tolist()[:241]
x_train = np.array(x_train)
y_train = np.array(y_train)

lr_model = LogisticRegression()
lr_model.fit(x_train, y_train)

y_pred = lr_model.predict(x_train)
print(y_pred)
print(y_train)
print("Accuracy on training set:", lr_model.score(x_train, y_train))