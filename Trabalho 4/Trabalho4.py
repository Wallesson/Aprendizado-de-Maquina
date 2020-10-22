"""
Trabalho 4

Nome: Wallesson Cavalcante da Silva
Matrícula: 397670

"""

import numpy as np
import matplotlib.pyplot #as plt 
from random import randrange, uniform
from sklearn.linear_model import LogisticRegression

class K_means():
    def __init__(self):
        pass
    def dis_euclidiana(self,centro, X):
        n = X.shape[0]
        m = centro.shape[0]
        X_ = np.zeros((n,m))
        
        for i in range(m):
            for j in range(n):
                a = X[i] - X[j]
                b = sum(a*a)
                X_[j][i] = np.sqrt(b)
        return X_

            
    def normalizer(self,X):
        xmin = X.min()
        xmax = X.max()
        return(X-xmin)/(xmax-xmin)
        
    def k_means(self, X, k):
        X_ = self.normalizer(X)
        centro = np.zeros((k,4))
        for i in range(k):
            centro[i] = X_[randrange(0, 150)]

        dist = self.dis_euclidiana(centro, X_) 

        self.b = np.zeros(X_.shape[0])
        
        for e in range(20):       
            for i in range(dist.shape[0]):
                aux = dist[i][0]
                t = 0
                for j in range(dist.shape[1]):
                    if(aux>dist[i][j]):
                        aux = dist[i][j]
                        t = j
                self.b[i] = t + 1
            for i in range(k):
                c = 0
                for j in range(self.b.shape[0]):
                    if(self.b[j] == i+1):
                        c +=1
                v = np.zeros((c,X_.shape[1]))
                g = 0
                for j in range(self.b.shape[0]):
                    if(self.b[j] == i+1):
                        v[g] = X_[j]
                        g+=1
                centro[i] = sum(v)/c
        
        for i in range(dist.shape[0]):
            aux = dist[i][0]
            t = 0
            for j in range(dist.shape[1]):
                if(aux>dist[i][j]):
                    aux = dist[i][j]
                    t = j
            self.b[i] = t + 1
        for i in range(k):
            c = 0
            for j in range(self.b.shape[0]):
                if(self.b[j] == i+1):
                    c +=1
            v = np.zeros((c,X_.shape[1]))
            g = 0
            for j in range(self.b.shape[0]):
                if(self.b[j] == i+1):
                    v[g] = X_[j]
                    g+=1

            a=self.dis_euclidiana(centro[i],v)
            matplotlib.pyplot.plot(a) 
            matplotlib.pyplot.show()            



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
           
class PCA():
    def __init__(self):
        pass
    def normalizer(self,X):
        xmin = X.min()
        xmax = X.max()
        return(X-xmin)/(xmax-xmin)
        
    def pca(self,X,y):
        X = self.normalizer(X)
        n = X.shape[0]
        media = sum(X)/n
        self.cov = []
        for i in range(n):
            a = X - media
            self.cov = a.T.dot(a)/(n-1)
        autovalores, autovetores = np.linalg.eig(self.cov)        
        pares = [(np.abs(autovalores[i]),autovetores[:,i]) for i in range(len(autovalores))]
        pares.sort()
        pares.reverse()
       
        componentes = 2
        autovetores = [p[1] for p in pares]
        a = autovetores[0:componentes]
        X = np.dot(X,np.array(a).T)
        total = sum(autovalores)
        var = [(i/total)*100 for i in sorted(autovalores, reverse=True)]
        print(var)
        ds = Dispersao()
        ds.plot_boundaries(X,y,y)
        
        
from sklearn import tree
from sklearn.model_selection import KFold


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


class Arvore_decisao():
    def __init__(self):
        pass
    def normalizer(self,X):
        xmin = X.min()
        xmax = X.max()
        return(X-xmin)/(xmax-xmin)
        
    def arvore_decisao_fit(self,X,y):
        X = self.normalizer(X)
        self.model = tree.DecisionTreeClassifier(criterion='gini') 
        self.model.fit(X, y)
        self.model.score(X, y)
        
    def arv_pred(self, X):
        return self.model.predict(X)

class K_FOLD():
    def __init__(self):
        pass
    def k_fold(self,X,y,k):
        
        b = int(X.shape[0]/k)
        v = b
        c = 0
        d = 0
        self.y_pred = []
        acv = np.zeros(5)
        
        for i in range(k):
            
            if(d<5):
                X_ts = X[c:b]
                y_ts = y[c:b]
                X_tr = np.concatenate((X[:c], X[b:]),axis=0)
                y_tr = np.concatenate((y[:c], y[b:]),axis=0)    
                a = Arvore_decisao()
                a.arvore_decisao_fit(X_tr,y_tr)
                pred = a.arv_pred(X_ts)
                ac = AC()
                acv[i] = ac.acuracia(y_ts,pred)
   
        print(sum(acv)/k)

        
    
        
data = np.loadtxt("trab4.data", skiprows=1, delimiter=',')
np.random.shuffle(data)
X = data[:, :4]
y = data[:, -1]

#f = K_means()
#f.k_means(X,5)
#p = PCA()
#p.pca(X,y)
k = K_FOLD()
k.k_fold(X,y,5)


