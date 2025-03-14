import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

NUM_CENTROIDS=8

data = pd.read_csv('synthetic.txt', delimiter=' ', header=None, names=['X','Y'])

X = data['X']
Y = data['Y']
print(X)
print(Y)
#randomly generate centroid indexes
C = np.random.choice(np.shape(X)[0], NUM_CENTROIDS)
print(C)

Distance = np.zeros((np.shape(X)[0], NUM_CENTROIDS))

epochs = 1

for i in epochs:
    for j in range(NUM_CENTROIDS):
        Distance[:,j] = np.sqrt((X-C[j])**2 + (Y-C[j])**2)
    print(Distance)




plt.xlabel('x')
plt.xlabel('y')
plt.title('Clustering')

#plot data (black) and centroids (yellow)
plt.scatter(X,Y)
plt.scatter(X[C],Y[C],c='y')
plt.show()