import matplotlib.pyplot as plt
import os
import numpy as np
import gzip

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def load_mnist(path, kind='train'):
    """Load MNIST data from 'path'"""
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' %kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' %kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        lbpath.read(8)
        buffer = lbpath.read()
        labels = np.frombuffer(buffer, dtype=np.uint8)
    with gzip.open(images_path, 'rb') as imgpath:
        imgpath.read(16)
        buffer = imgpath.read()
        images = np.frombuffer(buffer, dtype=np.uint8).reshape(len(labels), 28, 28).astype(np.float64)

    return images, labels

def vector_hoa(X):
    return X[:].reshape(-1, X[0].size)

def sampling(X):
    t = X.shape[0];
    n = range(t);
    m = range(0,28,2)
    Xt = np.zeros((t,14,14), dtype=int)
    for k in n:
        for i in m:
            for j in m:
                Xt[k][int(i/2)][int(j/2)] = (X[k][i][j] + X[k][i+1][j] + X[k][i][j+1] + X[k][i+1][j+1])/4
    return vector_hoa(Xt)

def histogram(X):
    t = X.shape[0];
    n = range(t);
    m = range(28)
    Xh = np.zeros((t,256), dtype=int)
    for k in n:
        for i in m:
            for j in m:
                Xh[k][int(X[k][i][j])]+=1
    return Xh

def MPL(X, y):
    d = np.zeros(10)
    x = np.zeros((10, X[0].size), dtype=int)
    n = range(X.shape[0])
    for k in n:
        x[y[k]] = x[y[k]] + X[k]
        d[y[k]] += 1
    
    m = range(x[0].size)
    for k in range(10):
        for i in m:
            x[k][i] = round(x[k][i] / d[k])
    return x

def suydoan(X, x):
    y = np.zeros(X.shape[0])
    n = range(X.shape[0])
    from scipy.spatial import distance

    for k in n:
        t = 0
        nn = distance.euclidean(X[k], x[0])
        for i in range(10):
            tam = distance.euclidean(X[k], x[i])
            if tam < nn:
                t = i
                nn = tam
        y[k] = t
    return y

def tinh(X, x, Y, y):
    tam1 = 0
    for i in range(1,10,2):
        model = KNeighborsClassifier(n_neighbors=i)
        model.fit(X, Y)
        y_pred = model.predict(x)
        tam = 100 * accuracy_score(y, y_pred)
        if tam1 < tam:
            tam1 = tam
    
    cls = MPL(X, Y)
    y_pred = suydoan(x, cls)

    tam2 = 100*accuracy_score(y, y_pred)
    return tam1, tam2

print('[INFO] Dang nap du lieu...');
X_train, y_train = load_mnist('data/', kind='train')
X_test, y_test = load_mnist('data/', kind='test')

print('[INFO] Nap du lieu hoan tat! Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]));

fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True,)
ax = ax.flatten()
for i in range(10):
    img = X_train[y_train == i][0]
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

print('[INFO] Dang vecto hoa...');
Xv_train = vector_hoa(X_train)
Xv_test = vector_hoa(X_test)
print('[INFO] Vecto hoa hoan tat');

print('[INFO] Dang thuc hien Sampling...');
Xs_train = sampling(X_train)
Xs_test = sampling(X_test)
print('[INFO] Sampling hoan tat');

print('[INFO] Dang thuc hien Histogram...');
Xh_train = histogram(X_train)
Xh_test = histogram(X_test)
print('[INFO] Histogram hoan tat');

bang = np.zeros((2, 3))

print('[INFO] Dang tinh toan...');
bang[0][0], bang[1][0] = tinh(Xv_train, Xv_test, y_train, y_test)
bang[0][1], bang[1][1] = tinh(Xs_train, Xs_test, y_train, y_test)
bang[0][2], bang[1][2] = tinh(Xh_train, Xh_test, y_train, y_test)
print('[INFO] Tinh toan hoan tat');

print(' %20s %14s %14s'%('vector','sampling','histogram'))
print('KNN%17.2f%%'%(bang[0][0]),end='');

for j in range(1, 3):
    print("%14.2f%%" %(bang[0][j]), end = '')
print('')
print('Mau phan lop%8.2f%%'%(bang[1][0]),end='');

for j in range(1, 3):
    print("%14.2f%%" %(bang[1][j]), end = '')
