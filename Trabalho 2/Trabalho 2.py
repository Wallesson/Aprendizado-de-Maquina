"""
Nome: Wallesson Cavalcante da Silva

Matrícula: 397670

"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn import datasets

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix

class R_Logistica_Gradiente(object):
    def __init__(self, alpha=0.01, epochs=1000):
        self.alpha = alpha
        self.epochs = epochs

    def fit(self, X, y):
        n = X.shape[0]
        m = X.shape[1]
        X_ = np.c_[np.ones(n), X]
        self.w = np.zeros(m+1)
        
        for i in range(self.epochs):            
            y_pred = 1/(1 + (np.exp(X_ @ -self.w)))
            erro = y - y_pred
            gradiente  = np.sum(erro @ X_)/n
            self.w += self.alpha * gradiente
   
    def predict(self,X):
        n = X.shape[0]
        X_ = np.c_[np.ones(n), X]
        a = 1/ (1 + (np.exp(X_ @ -self.w)))
        for i in range(n):
            if(a[i] >0.5):
                a[i] = 1
            else:
                a[i] = 0
        return a

class NB_Gaus(object):
    def __init__(self):
        pass
    
    def fit(self, X,y):
        n = X.shape[0]
        c1 = 0
        c2 = 0
        
        for i in range(n):
            if(y[i]==0):
                c1 +=1
            if(y[i]==1):
                c2 +=1
        x1_1 = np.zeros(c1)
        x2_1 = np.zeros(c1)
        y1 = np.zeros(c1)
        x1_2 = np.zeros(c2)
        x2_2 = np.zeros(c2)
        y2 = np.zeros(c2)
        X1 = X[:,:1]
        X2 = X[:,1:2]
        k= 0
        l= 0
        for i in range(n):
            if(y[i]==0):
                x1_1[k]=X1[i]
                x2_1[k]=X2[i]
                y1[k]=y[i]
                k+=1
            if(y[i]==1):
                x1_2[l]=X1[i]
                x2_2[l]=X2[i]
                y2[l]=y[i]
                l+=1
        X_1 = np.c_[x1_1,x2_1]
        X_2 = np.c_[x1_2,x2_2]
       
        self.pc1 = c1/n
        self.pc2 = c2/n
        
        self.miC1_x1 = np.mean(x1_1) 
        self.miC1_x2 = np.mean(x2_1)
        self.miC1 = np.c_[self.miC1_x1,self.miC1_x2]
       
        self.miC2_x1 = np.mean(x1_2)
        self.miC2_x2 = np.mean(x2_2)
        self.miC2 = np.c_[self.miC2_x1,self.miC2_x2]
        
        x = X_1 - self.miC1    
        self.MC1 = (x.T@x)/n
        
        x = X_2 - self.miC2
        self.MC2 = (x.T@ x)/n
    
        for i in range(2):
            for j in range(2):
                if(i != j):
                    self.MC1[i][j] = 0
                    self.MC2[i][j] = 0
    
   
    def predict(self, X):
        n = X.shape[0]
        det1 = np.linalg.det(self.MC1)
        det2 = np.linalg.det(self.MC2)
        
        inv1= np.linalg.inv(self.MC1)
        inv2= np.linalg.inv(self.MC1)
        
        p1 = det1**(1/2) * (2*np.pi**(2/2)) 
        p2 = det2**(1/2) * (2*np.pi**(2/2)) 
        
        self.pxc1 = np.zeros(n)
        self.pxc2 = np.zeros(n)
        
        for i in range(n):
            a = X[i]
            b = a - self.miC1
            c = (1/2) * b.T
            d = b @ inv1
            e = d @ c
            self.pxc1[i] = (1/p1) * np.exp(-e)
        
        for i in range(n):
            a = X[i]
            b = a - self.miC2
            c = (1/2) * b.T
            d = b @ inv2
            e = d @ c
            self.pxc2[i] = (1/p2) * np.exp(-e)
        
        self.pcx1 = self.pxc1 * self.pc1
        self.pcx2 = self.pxc2 * self.pc2
        
        y_pred = np.zeros(n)
        
        for i in range(n):
            if(self.pcx1[i]>self.pcx2[i]):
                y_pred[i] = 0
            else:
                y_pred[i] = 1

        return y_pred


class Discriminante_Qua_Gaus(object):
    def __init__(self):
        pass
    
    def fit(self, X,y):
        n = X.shape[0]
        c1 = 0
        c2 = 0
        
        for i in range(n):
            if(y[i]==0):
                c1 +=1
            if(y[i]==1):
                c2 +=1
        x1_1 = np.zeros(c1)
        x2_1 = np.zeros(c1)
        y1 = np.zeros(c1)
        x1_2 = np.zeros(c2)
        x2_2 = np.zeros(c2)
        y2 = np.zeros(c2)
        X1 = X[:,:1]
        X2 = X[:,1:2]
        k= 0
        l= 0
        for i in range(n):
            if(y[i]==0):
                x1_1[k]=X1[i]
                x2_1[k]=X2[i]
                y1[k]=y[i]
                k+=1
            if(y[i]==1):
                x1_2[l]=X1[i]
                x2_2[l]=X2[i]
                y2[l]=y[i]
                l+=1
        X_1 = np.c_[x1_1,x2_1]
        X_2 = np.c_[x1_2,x2_2]
       
        self.pc1 = c1/n
        self.pc2 = c2/n
        
        self.miC1_x1 = np.mean(x1_1) 
        self.miC1_x2 = np.mean(x2_1)
        self.miC1 = np.c_[self.miC1_x1,self.miC1_x2]
       
        self.miC2_x1 = np.mean(x1_2)
        self.miC2_x2 = np.mean(x2_2)
        self.miC2 = np.c_[self.miC2_x1,self.miC2_x2]
        
        x = X_1 - self.miC1    
        self.MC1 = (x.T@x)/n
        
        x = X_2 - self.miC2
        self.MC2 = (x.T@ x)/n
    
    def predict(self, X):
        n = X.shape[0]
        det1 = np.linalg.det(self.MC1)
        det2 = np.linalg.det(self.MC2)
        
        inv1= np.linalg.inv(self.MC1)
        inv2= np.linalg.inv(self.MC1)
        
        p1 = det1**(1/2) * (2*np.pi**(2/2)) 
        p2 = det2**(1/2) * (2*np.pi**(2/2)) 
        
        self.pxc1 = np.zeros(n)
        self.pxc2 = np.zeros(n)
        
        for i in range(n):
            a = X[i]
            b = a - self.miC1
            c = (1/2) * b.T
            d = b @ inv1
            e = d @ c
            self.pxc1[i] = (1/p2) * np.exp(-e)
        
        for i in range(n):
            a = X[i]
            b = a - self.miC2
            c = (1/2) * b.T
            d = b @ inv2
            e = d @ c
            self.pxc2[i] = (1/p1) * np.exp(-e)
        
        self.pcx1 = self.pxc1 * self.pc1
        self.pcx2 = self.pxc2 * self.pc2
        
        y_pred = np.zeros(n)
        
        for i in range(n):
            if(self.pcx1[i]>self.pcx2[i]):
                y_pred[i] = 0
            else:
                y_pred[i] = 1

        return y_pred

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
    
class Dispersao(object):
    def __init__(self):
        pass
    
    def plot_boundaries(self,X, Y, clf):
        r = R_Logistica_Gradiente()
        r.fit(X,Y)
        
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5     
        h = .02  
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = r.predict(np.c_[xx.ravel(), yy.ravel()])

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

class Confusao(object):
    def __init__(self):
        pass
    
    def plot_confusion_matrix(self,X_, y_, clf):
        iris = datasets.load_iris()
        X = X_
        y = y_
        class_names = iris.target_names
                
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        
        classifier = svm.SVC(kernel='linear', C=0.01).fit(X_train, y_train)
        
        np.set_printoptions(precision=2)

        titles_options = [("Confusion matrix, without normalization", None),
                          ("Normalized confusion matrix", 'true')]
        for title, normalize in titles_options:
            disp = plot_confusion_matrix(classifier, X_test, y_test,
                                         display_labels=class_names,
                                         cmap=plt.cm.Blues,
                                         normalize=normalize)
            disp.ax_.set_title(title)
        
            print(title)
            print(disp.confusion_matrix)
        
        plt.show()
    
data = np.loadtxt("ex2data1.txt", skiprows=1, delimiter=',')
np.random.shuffle(data)

porc = round(data.shape[0]*0.7)

X_treino = data[0:porc, :2]
y_treino = data[0:porc, -1]

X_teste = data[porc:, 0:2]
y_teste = data[porc:, -1]

a = AC()
c = Confusao()
ds = Dispersao()

r = R_Logistica_Gradiente()
r.fit(X_treino,y_treino)
y_pred = r.predict(X_teste)


print("Regeção Logistica Gradiente: ",a.acuracia(y_teste,y_pred))
ds.plot_boundaries(X_teste,y_teste,y_pred)
c.plot_confusion_matrix(X_teste,y_teste,y_pred)

#------------------------------------------------------
n = NB_Gaus()
n.fit(X_treino,y_treino)
y_pred = n.predict(X_teste)

print("Naive Bayes Gaussiano: ",a.acuracia(y_teste,y_pred))
ds.plot_boundaries(X_teste,y_teste,y_pred)
c.plot_confusion_matrix(X_teste,y_teste,y_pred)

#------------------------------------------------------
d = Discriminante_Qua_Gaus()
d.fit(X_treino, y_treino)
y_pred = d.predict(X_teste)

print("Discriminante Quadratico Gaussiano: ",a.acuracia(y_teste,y_pred))
ds.plot_boundaries(X_teste,y_teste,y_pred)
c.plot_confusion_matrix(X_teste,y_teste,y_pred)
