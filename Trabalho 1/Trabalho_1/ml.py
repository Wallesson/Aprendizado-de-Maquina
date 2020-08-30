"""
Nome: Wallesson Cavalcante da Silva

Matrícula: 397670

"""
import numpy as np
import matplotlib.pyplot as plt

class RL_Simples_Analitico(object):
    def __init__(self):
        pass
        
    def fit(self, X,Y):
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
        print("B0 simples analitico", self.B0)
        print("B1 simples analitico",self.B1)

        
    def predict(self, X):
        return self.B0+(self.B1*X)
   
class RL_Simples_Gradiente_D(object):
    def __init__(self):
        pass
        
    def fit(self, X, y):
        X_ = np.array(X)
        Y_ = np.array(y)
    
        self.B0 = 0
        self.B1 = 0
        a = 0.01
        e = 0
        
        while(e < 20):
            mul = X_*self.B1
            mul = Y_- mul
            mul = mul-self.B0  
            c = mul
            d = mul*X_
            c = c.mean()
            d = d.mean()
            self.B0 = self.B0+c*a
            self.B1 = self.B1+d*a
            e = e+1
        print("B0 simples gradiente ", self.B0)
        print("B1 simples gradiente ",self.B1)
            
    def predict(self, X):
        return self.B0+(self.B1*X)
    
    
class RL_Mult_Analitico():
    def __init__(self):
        self.b=None

    def fit(self, X, y):
        X_ = np.c_[np.ones(X.shape[0]), X]

        self.b = np.linalg.pinv(X_.T @ X_) @ X_.T @ y
        print(self.b)
        
    def predict(self, X):
        X_ = np.c_[np.ones(X.shape[0]), X]

        return X_ @ self.b
 
class RL_Mul_Gradiente_D():
    def __init__(self):
        pass
    
    def fit(self, X,y):
        self.X = np.array(X)
        self.y = np.array(y)
        n = self.X.shape[0]
        X_ = np.c_[np.ones(n), self.X]
        a = 0.01
        e = 0
        self.B = np.zeros(5) 
       
        while(e < 20):
            i=0
            k=0
            for i in range(self.B.shape[0]):
                for k in range(self.B.shape[0]):
                    b =self.B[i] * X_[i][k]
                    b = b - self.B[0]
                    b = self.y[i] - b
                    b = b* X_[i][k]
                    b = b.mean()
                self.B[i] = a * b
            e = e+1
        print("Beta: ",self.B)
    def predict(self,X):
        n = X.shape[0]
        X_ = np.c_[np.ones(n), X]
        m = 0
        k = 0
        r = 0
        y_ = np.zeros(n)
        x_ = np.zeros(self.B.shape[0])
        for m in range(n):
            for k in range(self.B.shape[0]):
                l=0
                for r in range(self.B.shape[0]):
                    x_[l]= X_[k][r]
                    l = l +1
            y_[m] = self.B.T @ x_

        return y_
    
class RL_Mul_Estocastico():
    def __init__(self):
        pass
    
    def fit(self, X,y):
        self.X = np.array(X)
        self.y = np.array(y)
        n = self.X.shape[0]
        X_ = np.c_[np.ones(n), self.X]
        a = 0.01
        e = 0
        self.B = np.zeros(5) 
       
        while(e < 20):
            i=0
            k=0
            for i in range(self.B.shape[0]):
                for k in range(self.B.shape[0]):
                    b =self.B[i] * X_[i][k]
                    b = b - self.B[0]
                    b = self.y[i] - b
                    b = b* X_[i][k]
                self.B = self.B + (a * b)
            e = e+1
        print("Beta: ",self.B)
    def predict(self,X):
        n = X.shape[0]
        X_ = np.c_[np.ones(n), X]
        m = 0
        k = 0
        r = 0
        y_ = np.zeros(n)
        x_ = np.zeros(self.B.shape[0])
        for m in range(n):
            for k in range(self.B.shape[0]):
                l=0
                for r in range(self.B.shape[0]):
                    x_[l]= X_[k][r]
                    l = l +1
            y_[m] = self.B.T @ x_

        return y_
    
class RLM_Quadratica():
    def __init__(self):
        pass
    
    def fit(self,X,y):
        a = X*X
        X_ = np.c_[X,a]
        self.R = RL_Mult_Analitico()
        self.R.fit(X_,y)
    
    def predict(self, X):
        a = X*X
        X_ = np.c_[X,a]
        return self.R.predict(X_)

class RLM_Cubica():
    def __init__(self):
        pass
    
    def fit(self, X,y):
        a = X*X
        b = a * X
        X_ = np.c_[X,a]
        X_ = np.c_[X_,b]
        self.RLM = RL_Mult_Analitico()
        self.RLM.fit(X_,y)
    
    def predict(self, X):
        a = X*X
        b = a * X
        X_ = np.c_[X,a]
        X_ = np.c_[X_,b]
        return self.RLM.predict(X_)

class Regularizacao():
    def __init__(self):
        pass
    def fit(self,X,y):
        self.X = np.array(X)
        self.y = np.array(y)
        n = self.X.shape[0]
        X_ = np.c_[np.ones(n), self.X]
        a = 0.01
        e = 0
        self.B = np.zeros(5) 
        lamb = 1
        while(e < 20):
            i=0
            k=0
            for i in range(self.B.shape[0]):
                for k in range(self.B.shape[0]):
                    b =self.B[i] * X_[i][k]
                    b = b - self.B[0]
                    b = self.y[i] - b
            self.B[0] = self.B[0] + (a * b)
            for i in range(self.B.shape[0]):
                for k in range(self.B.shape[0]):
                    b =self.B[i] * X_[i][k]
                    b = b - self.B[0]
                    b = self.y[i] - b
                    b = b* X_[i][k]
                    lamb = (lamb/self.X.shape[0]) * self.B[i]
                    b = b-lamb
                    b = b.mean()
                    b = a * b
                    self.B[i] = self.B[i]+b
            e = e+1
            
    def predict(self,X):
        n = X.shape[0]
        X_ = np.c_[np.ones(n), X]
        m = 0
        k = 0
        r = 0
        y_ = np.zeros(n)
        x_ = np.zeros(self.B.shape[0])
        for m in range(n):
            for k in range(self.B.shape[0]):
                l=0
                for r in range(self.B.shape[0]):
                    x_[l]= X_[k][r]
                    l = l +1
            y_[m] = self.B.T @ x_

        return y_
    
class Fun():
    def __init__(self):
        pass
    
    def TSS(self,Y,teste_Pred):
        Media_Y = np.mean(teste_Pred)
        return np.sum((Y-Media_Y)**2)
       
    def RSS(self,Y,teste_Pred):
        Dif_Y_teste = Y-teste_Pred
        return np.sum(Dif_Y_teste**2)
    
    def F_MSE (self,Y,MEDV_pred):
        Sub_ = Y - MEDV_pred
        Mult= Sub_ * Sub_
        SOMA = sum(Mult)
        MSE_ = SOMA.mean()
        return MSE_
    
    def F_R2(self,Y,teste_Pred):
        return 1-(self.RSS(Y, teste_Pred)/self.TSS(Y, teste_Pred))
    
data = np.loadtxt("housing.data")
np.random.shuffle(data)

porc = round(data.shape[0]*0.8)

X_Treino = data[0:porc, -2]
y_Treino = data[0:porc, -1]

X_Teste = data[porc:, -2]
y_Teste = data[porc:, -1]

f = Fun()
print("RLS Analitico")
rsa = RL_Simples_Analitico()
rsa.fit(X_Treino,y_Treino)
y_pred = rsa.predict(X_Teste)

print("MSE Teste : ",f.F_MSE(y_Teste,y_pred))
print("R2 Teste: ",f.F_R2(y_Teste,y_pred))

y_pred = rsa.predict(X_Treino)

print("MSE Treino : ",f.F_MSE(y_Treino,y_pred))
print("R2 Treino : ",f.F_R2(y_Treino,y_pred))
#-----------------
print("RLS Gradiente")
rsg = RL_Simples_Gradiente_D()
rsg.fit(X_Treino,y_Treino)
y_pred_g = rsg.predict(X_Teste)

print("MSE Teste : ",f.F_MSE(y_Teste,y_pred_g))
print("R2 Teste: ",f.F_R2(y_Teste,y_pred_g))

y_pred_g = rsg.predict(X_Treino)

print("MSE Treino : ",f.F_MSE(y_Treino,y_pred_g))
print("R2  Treino: ",f.F_R2(y_Treino,y_pred_g))
#-----------------
print("RLM Quadratica")
rlmq = RLM_Quadratica()
rlmq.fit(X_Treino,y_Treino)
y_p = rlmq.predict(X_Teste)

print("MSE Teste : ",f.F_MSE(y_Teste,y_p))
print("R2 Teste : ",f.F_R2(y_Teste,y_p))

y_p = rlmq.predict(X_Treino)

print("MSE Treino : ",f.F_MSE(y_Treino,y_p))
print("R2 Treino : ",f.F_R2(y_Treino,y_p))

#-----------------
print("RLM Cubica")
rlmc = RLM_Cubica()
rlmc.fit(X_Treino,y_Treino)
y_pc = rlmc.predict(X_Teste)

print("MSE Teste : ",f.F_MSE(y_Teste,y_pc))
print("R2 Teste : ",f.F_R2(y_Teste,y_pc))

y_pct = rlmc.predict(X_Treino)

print("MSE Treino : ",f.F_MSE(y_Treino,y_pct))
print("R2 Treino : ",f.F_R2(y_Treino,y_pct))

data2 = np.loadtxt("trab1_data.txt")

porc = round(data2.shape[0]*0.8)

X_tre = data2[0:porc, 0:-2]
y_tre = data2[0:porc, -1]

X_tes = data2[porc:, 0:-2]
y_tes = data2[porc:, -1]

rlma = RL_Mult_Analitico()
rlmd = RL_Mul_Gradiente_D()
rlme = RL_Mul_Estocastico()
reg = Regularizacao()


print("RLM Analitico")
print("Beta: ")
rlma.fit(X_tre,y_tre)
y_pred = rlma.predict(X_tes)

print("MSE Treino: ", f.F_MSE(y_tes,y_pred))
print("R2 Treino: ", f.F_R2(y_tes,y_pred))
y_pred = rlma.predict(X_tre)

print("MSE Teste: ", f.F_MSE(y_tre,y_pred))
print("R2 Teste:", f.F_R2(y_tre,y_pred))
#--------------------------------------

print("RLM Gradiente Descendente")
rlmd.fit(X_tre,y_tre)
y_pred = rlmd.predict(X_tes)

print("MSE Treino: ",f.F_MSE(y_tes,y_pred))
print("R2 Treino: ",f.F_R2(y_tes,y_pred))

y_pred = rlmd.predict(X_tre)

print("MSE Teste: ",f.F_MSE(y_tre,y_pred))
print("R2 Teste: ",f.F_R2(y_tre,y_pred))
#-------------------------------------
print("RLM Estocastico")
rlme.fit(X_tre,y_tre)
y_pred = rlme.predict(X_tes)

print("MSE Treino: ",f.F_MSE(y_tes,y_pred))
print("R2 Treino: ",f.F_R2(y_tes,y_pred))

y_pred = rlme.predict(X_tre)

print("MSE Teste: ",f.F_MSE(y_tre,y_pred))
print("R2 Teste: ",f.F_R2(y_tre,y_pred))
#---------------------------------------
print("RLM Regularização")
reg.fit(X_tre,y_tre)
y_pred = reg.predict(X_tes)

print("MSE Treino: ",f.F_MSE(y_tes,y_pred))
print("R2 Treino: ",f.F_R2(y_tes,y_pred))

y_pred = reg.predict(X_tre)

print("MSE Teste: ",f.F_MSE(y_tre,y_pred))
print("R2 Teste: ",f.F_R2(y_tre,y_pred))
