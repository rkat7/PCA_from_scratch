import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt

data=np.array([[2,1],[3,4],[5,0],[7,6],[9,2]])
m1=0
m2=0
for i in data:
    m1+=i[0]
for i in data:
    m2+=i[1]

m1=m1/5
m2=m2/5

mean=[m1,m2]
z=abs(data-mean)
y=z.T

cov_y=np.cov(y)
eig_va,eig_ve=linalg.eig(cov_y)
print(eig_va,eig_ve)

t=np.dot(z,eig_ve)

plt.scatter(t[:,0],t[:,1])
plt.show()
