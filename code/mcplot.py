import matplotlib
matplotlib.use('Agg')
from matplotlib.pylab import *
from scipy import stats
NP=64
N=8
dim=2*NP+2
#title('alpha')
A=[[] for i in range(dim)]
Datafile=open("chain.dat",'r')
lines = Datafile.readlines()
i=0
j=0
for line in lines:
	i=i+1
	if i%1==0:
		j=j+1
		X = line.split()
		for w in range(dim):
			A[w].append(float(X[w+1]))
print(j)
Datafile=open("gamma.txt",'r')
gamma=[]
lines=Datafile.readlines()
for line in lines:
	real,imag=line.split()
	gamma.append((float(real),float(imag)))

t1=[]
t2=[]
for i in range(NP):
	temp1=0.0
	temp2=0.0
	num=0
	for q in A[2*i]:
		num+=1
		temp1+=q
	temp1=temp1/num
	for q in A[2*i+1]:
		temp2+=q
	temp2=temp2/num
	t1.append(temp1)
	t2.append(temp2)
t1=np.array(t1)
t2=np.array(t2)

savefig('data.png')
g1=np.array([gamma[i][0] for i in range(NP)])
g2=np.array([gamma[i][1] for i in range(NP)])
dg1=t1-g1
dg2=t2-g2
g1n=g1[np.abs(g1)<0.1]
g2n=g2[np.abs(g2)<0.1]
dg1n=dg1[np.abs(g1)<0.1]
dg2n=dg2[np.abs(g2)<0.1]


f=open("deltag1.txt",'a')
for i in range(len(g1n)):
	f.write("%f   %f\n" %(g1n[i],dg1n[i])
f.close()
f=open("deltag2.txt",'a')
for i in range(len(g1n)):
	f.write("%f   %f\n" %(g1n[i],dg1n[i])






