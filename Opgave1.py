import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

w = np.zeros(10) 
dataset = pd.read_csv("Dataset1.csv")
print(dataset)
w=np.zeros(10)
k_iter = 5
L_train=(len(dataset))
eta = 0.05


# for i in range(k_iter):
#     for j in range(L_train):
#         grad = de(k)dw
#         W = W +eta*grad
#     e(i) = mean(e^2)
L=100
k=np.arange(1,L,1)
t = np.arange(0,1,0.01)
y = np.sin(2*np.pi*t)

#plt.plot(y)
#plt.show()

nk = np.random.normal(0,np.sqrt(0.09),t.size)
tk = t
yk = np.sin(2*np.pi*tk)+nk


plt.plot(yk, 'o', color='black')
plt.show()