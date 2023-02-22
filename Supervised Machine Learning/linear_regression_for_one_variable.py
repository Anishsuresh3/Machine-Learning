import numpy as np
import matplotlib.pyplot as plt
import math

def compute(x,w,b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i]=w*x[i]+b;
    return f_wb

# computing the cost function J(w,b)
# returns the computed cost function for the given w and b
def compute_cost(x,y,w,b):
    m = x.shape[0]
    cost=0
    for i in range(m):
        tmp_f=w*x[i]+b
        cost+=round((tmp_f-y[i])**2,16)
    total_cost=cost/(2*m)
    return total_cost

#computing both w and b gradient with respect to w and b respectively
#and returns the scalar w and b gradients
def compute_gradient(x,y,w,b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        tmp_f = w*x[i]+b
        tmp_dj_dw = round((tmp_f-y[i])*x[i],16)
        tmp_dj_db = tmp_f-y[i]
        dj_dw+=tmp_dj_dw
        dj_db+=tmp_dj_db
    
    dj_dw = dj_dw/m
    dj_db = dj_db/m
    return (dj_dw,dj_db)


# updates the w and b parameters and finds the minimum cost function with respect to w and b
# return -
# - Updated value of w after running gradient descent 
# - Updated value of b after running gradient descent
# - return the history of cost values
# - returns the history of parameters [w,b]
def gradient_descent(x,y,w_in,b_in,alpha,iter):
    w=w_in
    b=b_in
    j_his=[]
    p_his=[]
    for i in range(iter):
        dj_dw,dj_db = compute_gradient(x,y,w,b)
        w = w - alpha*dj_dw
        b = b - alpha*dj_db
        if i<1000: # prevent resource exhaustion 
            j_his.append(compute_cost(x,y,w,b))
            p_his.append([w,b])
        
        if i% math.ceil(iter/10) == 0:
            print(f"Iteration {i:4}: Cost {j_his[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")
        
    
    return w,b,j_his,p_his

def main():
    # fig, ax = plt.subplots(1,1, figsize=(12, 6))
    # plt_contour_wgrad(x_train, y_train, p_hist, ax)
    x_train = np.array([1.1,1.3,1.5,2.0,2.2,2.9,3.0,3.2,3.2,3.7,3.9,4.0,4.0,4.1,4.5,4.9,5.1,5.3,5.9,6.0])
    y_train = np.array([39343,46205,37731,43525,39891,56642,60150,54445,64445,57189,63218,55794,56957,57081,61111,67938,66029,83088,81363,93940])
    # print(x_train.shape[0],y_train.shape[0])
    # x_train=np.array([158.0,166.0,163.0,165.0,167.0,170.0,167.0,172.0,177.0,181.0])# height of fathers
    # y_train=np.array([163.0,158.0,167.0,170.0,160.0,180.0,170.0,175.0,172.0,175.0]) # height of sons
    # x_train=x_train/100
    # y_train=y_train/100
    w_initial=0
    b_initial=0
    alpha=0.01
    iter=10000
    w_final,b_final,j_history,p_history=gradient_descent(x_train,y_train,w_initial,b_initial,alpha,iter)
    print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")
    # y=w_final*0.164+b_final
    # print("Height of son is ",y)
    fig,ax=plt.subplots()
    ax.plot(x_train,compute(x_train,w_final,b_final))
    scatter = ax.scatter(x_train,y_train,marker='X')
    plt.show()
    # If height of father is 164cm (x)
    # height of son is (y)

main()