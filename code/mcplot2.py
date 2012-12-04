import matplotlib
matplotlib.use('Agg')
from matplotlib.pylab import *
from scipy import stats
import numpy as np
g1n=[]
dg1n=[]
g2n=[]
dg2n=[]
Datafile=open("deltag1.txt",'r')
lines = Datafile.readlines()
for line in lines:
	x,y=line.split()
	g1n.append(float(x))
	dg1n.append(float(y))

Datafile=open("deltag2.txt",'r')
lines = Datafile.readlines()
for line in lines:
	x,y=line.split()
	g2n.append(float(x))
	dg2n.append(float(y))

g1n=np.array(g1n)
dg1n=np.array(dg1n)
g2n=np.array(g2n)
dg2n=np.array(dg2n)

fig = figure()
fig.add_subplot(111)
slope,intercept,r_value,p_value,std_err=stats.linregress(g1n,dg1n)
xlabel('g1')
ylabel('dg1')
plot(g1n,intercept+slope*g1n,'r')
scatter(g1n,dg1n)
savefig('g1.png')

fig = figure()
fig.add_subplot(111)
slope2,intercept2,r_value2,p_value2,std_err2=stats.linregress(g2n,dg2n)
xlabel('g2')
ylabel('dg2')
plot(g2n,intercept2+slope2*g2n,'r')
scatter(g2n,dg2n)
savefig('g2.png')
print(slope,intercept,r_value,p_value,std_err)
print(slope2,intercept2,r_value2,p_value2,std_err2)


bins = linspace(-0.1, 0.1, 21)
digitized = np.digitize(g1n, bins)
dg_means = [dg1n[digitized == i].mean() for i in range(1, len(bins))]
print(dg_means)



