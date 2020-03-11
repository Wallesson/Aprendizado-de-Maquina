import numpy as np
import matplotlib.pyplot as plt

base_dados = np.loadtxt('Advertising.csv', skiprows=1, delimiter=',')
X = base_dados[:,3]
Y = base_dados[:,4]

class Regressao_Linar_Simples(object):
    def __init__(self):
        self.B0 = None
        self.B1 = None
        
    def fit(self, a,b):
        self.X = np.array(X)
        self.Y = np.array(Y)
        
        Media_X = X.mean()
        Media_Y = Y.mean()
        
        Difer_X = X-Media_X
        Difer_Y = Y-Media_Y
        
        Difer_X_vezes_Y = Difer_X*Difer_Y
        Difer_X_vezes_X = Difer_X*Difer_X
        
        Superior = sum(Difer_X_vezes_Y)
        Inferior = sum(Difer_X_vezes_X)
        
        self.B1 = Superior/Inferior
        self.B0 = Media_Y-(self.B1* Media_X)

        print("Beta_0: ", self.B0)
        print("Beta_1: ", self.B1)    
    def predict(self, X):
        return self.B0+(self.B1*X)


teste = Regressao_Linar_Simples()
teste.fit(X,Y)
teste_Pred = teste.predict(X)

plt.scatter(X,Y)
plt.plot(X,teste_Pred, c='r')


class Funcoes():
    def __init__(self):
        pass
    
    def RSS(self,Y,teste_Pred):
        Dif_Y_teste = Y-teste_Pred
        return np.sum(Dif_Y_teste**2)
    
    def RSE(self,Y,teste_Pred):
       return np.sqrt((self.RSS(Y, teste_Pred))/(Y.size-2))
    
    def TSS(self,Y,teste_Pred):
       Media_Y = np.mean(teste_Pred)
       return np.sum((Y-Media_Y)**2)
   

    def R2(self,Y,teste_Pred):
       return 1-(self.RSS(Y, teste_Pred)/self.TSS(Y, teste_Pred))
   
    def MAE(self,Y,teste_Pred): 
       return np.sum(np.abs(Y - teste_Pred))/Y.size
   
funcao = Funcoes()
print("RSS_Simples:",funcao.RSS(Y,teste_Pred))
print("RSS_Simples:",funcao.RSE(Y,teste_Pred))
print("RSS_Simples:",funcao.R2(Y,teste_Pred))    
print("RSS_Simples:",funcao.MAE(Y,teste_Pred)) 
            
    