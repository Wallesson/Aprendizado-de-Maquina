import numpy as np
import matplotlib.pyplot as plt

class Fun():
    def TSS(self,Y,teste_Pred):
        Media_Y = np.mean(teste_Pred)
        return np.sum((Y-Media_Y)**2)
       
    def RSS(self,Y,teste_Pred):
        Dif_Y_teste = Y-teste_Pred
        return np.sum(Dif_Y_teste**2)
    
    def F_MSE (self,Y,MEDV_pred):
        Sub_ = Y-MEDV_pred
        Mult= Sub_ * Sub_
        SOMA = sum(Mult)
        MSE_ = SOMA.mean()
        return MSE_
    
    def F_R2(self,Y,teste_Pred):
        return 1-(self.RSS(Y, teste_Pred)/self.TSS(Y, teste_Pred))

  
class SimpleLinearRegression():
    def __init__(self):
        pass

    def fit(self, X, y):
        y_bar = np.mean(y)
        x_bar = np.mean(X)

        x_x_bar = X - x_bar
        y_y_bar = y - y_bar
        
        num = np.sum(x_x_bar * y_y_bar)

        denom = np.sum(x_x_bar * x_x_bar)

        self.b1 = num/denom
        self.b0 = y_bar - self.b1 * x_bar

    def predict(self, X):
        return self.b0 + X*self.b1



data = np.loadtxt("housing.data")
np.random.shuffle(data)

porc = round(data.shape[0]*0.8)

LSTAT_Treino = data[0:porc, -2]
MEDV_Treino = data[0:porc, -1]

LSTAT_Teste = data[porc:, -2]
MEDV_Teste = data[porc:, -1]

reg = SimpleLinearRegression()
reg.fit(LSTAT_Treino, MEDV_Treino)
MEDV_pred = reg.predict(LSTAT_Teste)

f = Fun()

MSE_Teste = f.F_MSE(MEDV_Teste,MEDV_pred)
R_2_Teste = f.F_R2(MEDV_Teste,MEDV_pred)

print("MSE Conjunto de Teste: ",MSE_Teste)
print("R2 Conjunto de Teste: ",R_2_Teste)

plt.scatter(LSTAT_Teste,MEDV_Teste)
plt.plot(LSTAT_Teste,MEDV_pred,c='r')
plt.show()

plt.scatter(MEDV_pred,MEDV_Teste)
plt.plot(MEDV_Teste,MEDV_Teste, c='r')
plt.show()

L_2_Treino = LSTAT_Treino * LSTAT_Treino
M_2_Treino = MEDV_Treino * MEDV_Treino

L_2_Teste = LSTAT_Teste * LSTAT_Teste
M_2_Teste = MEDV_Teste * MEDV_Teste
reg_2 = SimpleLinearRegression()
reg_2.fit(L_2_Treino, M_2_Treino)
MEDV_2_pred = reg_2.predict(L_2_Teste)


MSE_2_Teste = f.F_MSE(M_2_Teste,MEDV_2_pred)
R_2_Teste_2 = f.F_R2(M_2_Teste,MEDV_2_pred)

print("MSE_2 Conjunto de Teste: ",MSE_2_Teste)
print("R2_2 Conjunto de Teste: ",R_2_Teste_2)

plt.scatter(L_2_Teste,M_2_Teste)
plt.plot(L_2_Teste,MEDV_2_pred,c='r')
plt.show()

plt.scatter(MEDV_2_pred,M_2_Teste)
plt.plot(M_2_Teste,M_2_Teste, c='r')
plt.show()

L_3_Treino = LSTAT_Treino * LSTAT_Treino * LSTAT_Treino
M_3_Treino = MEDV_Treino * MEDV_Treino * MEDV_Treino

L_3_Teste = LSTAT_Teste * LSTAT_Teste * LSTAT_Teste
M_3_Teste = MEDV_Teste * MEDV_Teste * MEDV_Teste
reg_3 = SimpleLinearRegression()
reg_3.fit(L_3_Treino, M_3_Treino)
MEDV_3_pred = reg_3.predict(L_3_Teste)


MSE_3_Teste = f.F_MSE(M_3_Teste,MEDV_3_pred)
R_2_Teste_3 = f.F_R2(M_3_Teste,MEDV_3_pred)

print("MSE_3 Conjunto de Teste: ",MSE_3_Teste)
print("R2_3 Conjunto de Teste: ",R_2_Teste_3)

plt.scatter(L_3_Teste,M_3_Teste)
plt.plot(L_3_Teste,MEDV_3_pred,c='r')
plt.show()

plt.scatter(MEDV_3_pred,M_3_Teste)
plt.plot(M_3_Teste,M_3_Teste, c='r')
plt.show()