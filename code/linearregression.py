import numpy as np
C=np.linalg.inv(np.diag(v1))
Y=np.matrix(dg1).transpose()
A=[]
for i in range(len(g1)):
    A.append((1,g1[i]))
A=np.matrix(A)
M=np.linalg.inv(A.transpose()*C*A)*(A.transpose()*C*Y)
print('b,m')
print(M)

sig=np.linalg.inv(A.transpose()*C*A)
print('uncertainty')
print(sig)
    
