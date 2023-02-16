import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


"""
Takes in parameters-
x(nd array) - Shape (n,) example with multiple features
w(nd array) - Shape (n,) model parameters 
b - model parameter

Returns 
f_wb(scalar) - prediction
"""
def predict(x,w,b):
    f_wb=np.dot(x,w)+b
    return f_wb

"""
Takes in parameters-
X(ndarray (m,n)): Data, m examples with n features
y(ndarray (m,)) : target values
w(ndarray (n,)) : model parameters  
b(scalar) : model parameter

Returns
computed cost

"""
def compute_cost(x,y,w,b):
    m = len(x)
    cost=0.0
    for i in range(m):
        f_wbi=np.dot(x[i],w)+b
        cost+=(f_wbi-y[i])**2
    cost=cost/(2*m)
    return cost

def compute_gradient(x,y,w,b):
    m,n=len(x),len(x[0])
    dj_dw=np.zeros((n,))
    dj_db=0.
    for i in range(m):
        f_wbi=(np.dot(x[i],w)+b)-y[i]
        for j in range(n):
            dj_dw[j] += f_wbi*x[i][j]
        dj_db+=f_wbi
    dj_dw=dj_dw/m
    dj_db=dj_db/m
    
    return dj_dw,dj_db
        

def gradient_descent(x,y,w_in,b_in,alpha,iter):
    w=w_in
    b=b_in
    j_his=[]
    p_his=[]
    for i in range(iter):
        dj_dw,dj_db = compute_gradient(x,y,w,b)
        w = w-alpha*dj_dw
        b = b-alpha*dj_db

        if i%100==0: # prevent resource exhaustion 
            j_his.append(compute_cost(x,y,w,b))
        #     p_his.append([w,b])
        
        # if i% math.ceil(iter/10) == 0:
        #     print(f"Iteration {i:4}: ",
        #           f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
        #           f"w: {w: 0.3e}, b:{b: 0.5e}")
    
    return w,b,j_his,p_his

def main():
    # reading from excel and coverting to csv
    # read_file = pd.read_excel("Movie_dataset.xlsx")
    # read_file.to_csv("Movies.csv",index=None)
    df = pd.read_csv("Datasets/Movies.csv")
    # print(df)
    """
    In the dataset we have - 
    The data (X1, X2, X3, X4) are for each movie
    X1 = first year box office receipts/millions
    X2 = total production costs/millions
    X3 = total promotional costs/millions
    X4 = total book sales/millions 

    y = we'll consider X1 as target value that is by considering the total production, promotion and book sales 
    we'll predict the first year box office

    """
    x_train=[]
    for i in range(len(df)):
        x_train.append(df.loc[i].tolist()[1:])
    # x_train = np.array([[8.5, 5.099999905, 4.699999809], [12.89999962, 5.800000191, 8.800000191], [5.199999809, 2.099999905, 15.10000038], [10.69999981, 8.399998665, 12.19999981], [3.099999905, 2.900000095, 10.60000038], [3.5, 1.200000048, 3.5], [9.199999809, 3.700000048, 9.699999809], [9.0, 7.599999905, 5.900000095], [15.10000038, 7.699999809, 20.79999924], [10.19999981, 4.5, 7.900000095]])
    y_train = df['X1'].tolist()
    # y_train = np.array([85.09999847, 106.3000031, 50.20000076, 130.6000061, 54.79999924, 30.29999924, 79.40000153, 91.0, 135.3999939, 89.30000305])
    alpha=5.0e-3
    iter=10000
    w_initial=np.zeros(3)
    b_initial=0
    w_final,b_final,j_his,p_his=gradient_descent(x_train,y_train,w_initial,b_initial,alpha,iter)
    # print(f"b,w found by gradient descent: {b_final} , {w_final} ")
    print(w_final)
    print(b_final)
    fig,ax=plt.subplots()
    # ax.plot(x_train,predict(x_train,w_final,b_final))
    iter_arr=np.arange(100,10001,100)
    ax.plot(iter_arr,j_his)
    # scatter = ax.scatter(x_train,y_train,marker='X')
    # x=np.array([12.89999962, 5.800000191, 8.800000191])
    # y = np.dot(w_final,x)+b_final
    # print(y)
    plt.show()
#y=7.676+3.6616⋅x1+7.6211⋅x2+0.8285⋅x3

main()
    


    

# [[8.5, 5.099999905, 4.699999809], [12.89999962, 5.800000191, 8.800000191], [5.199999809, 2.099999905, 15.10000038], [10.69999981, 8.399998665, 12.19999981], [3.099999905, 2.900000095, 10.60000038], [3.5, 1.200000048, 3.5], [9.199999809, 3.700000048, 9.699999809], [9.0, 7.599999905, 5.900000095], [15.10000038, 7.699999809, 20.79999924], [10.19999981, 4.5, 7.900000095]]