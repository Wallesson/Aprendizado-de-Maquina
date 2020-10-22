'''
Nome: Wallesson Cavalcante da Silva
Matrícula: 397670
'''

import numpy as np
import matplotlib.pyplot as plt

import random
from sklearn.linear_model import LogisticRegression
from sklearn import datasets


class AC(object):
    def __init__(self):
        pass
    def acuracia(self, y, y_pred):
        n = y.shape[0]
        c = 0
        for i in range(n):
            if(y[i]==y_pred[i]):
                c+=1
        x = (c*100/n)/100
        return x
    
class KNN():
    def __init__(self):
        pass
    
    def fit_KNN(self, X, y):
        self.X_ = np.c_[X,y]

        
    def predict_KNN(self, k, X):
        n = X.shape[0]
        m = self.X_.shape[0]
        
        y_pred = np.zeros(n)
        self.t = np.zeros(m)
        a = self.X_
        lista = np.zeros(m)
        for i in range(m):
            lista[i] = i
        a = np.c_[lista,a]
        a = a[:, 1:3]
        for c in range(n):
            l = X[c]
            for i in range(m):
                self.t[i] = np.sqrt(sum((l-a[i])**2))
            
            f = np.c_[self.t,self.X_[:, 2:]]
            aux = 0
            aux2 = 0
            for i in range(f.shape[0]):
                for j in range(f.shape[0]):
                    if(f[i][0]<f[j][0]):
                        aux = f[i][0]
                        f[i][0] = f[j][0]
                        f[j][0] = aux
                        aux = 0
                        aux2 = f[i][1]
                        f[i][1] = f[j][1]
                        f[j][1] = aux2
                        aux2 = 0
            c1 = 0
            c2 = 0
            
            for h in range(k):  
                if(f[h][1] == 1):
                    c1 = c1 + 1
                else:
                    c2 = c2 + 1
            if(c1>c2):
                y_pred[c] = 1
            else:
                y_pred[c] = 0
                 
        return y_pred
                
class MLP():
    def __init__(self):
        pass
    def sig(self,x):
        a = 1/(1+np.exp(-x))
        return a
    def sigL(self,x):
        a = self.sig(x)*(1-self.sig(x))
        return a
    def normalizer(self,X):
        xmin = X.min()
        xmax = X.max()
        return(X-xmin)/(xmax-xmin)
        
    def fit(self, X, y,t):
        n = X.shape[0]
        #X = self.normalizer(X)
        X_ = np.c_[np.ones(n)*-1, X]
        X_ = self.normalizer(X_)
        n = X_.shape[0]
        m = X_.shape[1]
        self.w = np.random.rand(t, m)
        self.m = np.random.rand(1, t+1)
        self.alpha = 0.01
        for o in range(100):
            for i in range(n):
                Ui = X_[i]@self.w.T
                
                Zi = np.zeros(Ui.shape[0])
                for j in range(Ui.shape[0]):
                    Zi[j] = self.sig(Ui[j])
                
                a = np.append(-1,Zi)
                for j in range(a.shape[0]):
                    a[j] = a[j]*self.m[0][j]
                Uk = sum(a)
                
                yk = self.sig(Uk)
    
                if(yk>0.5):
                    yk=1
                else:
                    yk=0
                
                Ek = y[i] - yk
                
                deltak = Ek*self.sigL(Uk)
                
                deltai = np.zeros(Ui.shape[0])
                
                for j in range(deltai.shape[0]):
                    deltai[j] = self.sigL(Ui[j])*(deltak*self.m[0][j])
                
                c = self.alpha*deltak
                
                for j in range(self.w.shape[0]):
                    for k in range(self.w.shape[1]):
                        self.w[j][k] = self.w[j][k] + (c * X_[0][k])
                
                for j in range(self.m.shape[1]-1):
                    self.m[0][j] = self.m[0][j] + c*Zi[j]
            
            
    def predict(self,X,t):
        n = X.shape[0] 

        X_ = np.c_[np.ones(n)*-1, X]
        X_ = self.normalizer(X_)
        n = X_.shape[0]
        m = X_.shape[1]
                
        y_pred = np.zeros(n)
        for i in range(n):
            Ui = X_[i]@self.w.T
            Zi = np.zeros(Ui.shape[0])
            for j in range(Ui.shape[0]):
                Zi[j] = self.sig(Ui[j])
            a = np.append(-1,Zi)
            
            for j in range(a.shape[0]):
                a[j] = a[j]*self.m[0][j]
            Uk = sum(a)
            yk = self.sig(Uk)
            if(yk>0.5):
                y_pred[i]=1
            else:
                y_pred[i]=0
        
        return y_pred
                
class Dispersao(object):
    def __init__(self):
        pass
    
    def plot_boundaries(self,X, Y, clf):
        logreg = LogisticRegression(C=1e5)
        logreg.fit(X, Y)
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        h = .02
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.figure(1, figsize=(4, 3))
        plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
        plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
        plt.xlabel('Sepal length')
        plt.ylabel('Sepal width')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.show()
        
class K_FOLD():
    def __init__(self):
        pass
    def k_fold(self,X,y,k,metodo):
        
        b = int(X.shape[0]/k)
        v = b
        c = 0
        d = 0
        self.y_pred = []
        a = np.zeros(5)
        
        for i in range(k):
            
            if(d<5):
                X_ts = X[c:b]
                y_ts = y[c:b]
                X_tr = np.concatenate((X[:c], X[b:]),axis=0)
                y_tr = np.concatenate((y[:c], y[b:]),axis=0)    
               
                if(metodo==1):
                    m = MLP()
                    m.fit(X_tr,y_tr,4)
                    self.y_pred = m.predict(X_ts,4)
                                        
                else:
                    knn = KNN()
                    knn.fit_KNN(X_tr, y_tr)
                    self.y_pred = knn.predict_KNN(3,X_ts)
                    
                ac = AC()
                a[i] = ac.acuracia(y_ts,self.y_pred)
                c = b
                d = d+1
                b += v 
        f = sum(a)

        print(f/k)
        ds = Dispersao()
        ds.plot_boundaries(X_ts,y_ts,self.y_pred)
        
                  
data = np.loadtxt("data1.txt", skiprows=1, delimiter=',')
np.random.shuffle(data)

X = data[:, :2]
y = data[:, -1]

k = K_FOLD()
k.k_fold(X,y,5,1)

