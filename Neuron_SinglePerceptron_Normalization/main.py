from FunkcjeNeuronu import *
import pandas as pd
from Normalizacja1_0 import multiply, Normalization

x=[[4,8,1],[0.2,1,6],[7,5,2.1],[0.9,2.4,10],[13,17,3],[1.5,1.9,6],[10,3.1,2.4],[2,2,0.1],[1.9,2.9,2.1]]
y = [1,0,1,0,1,0,1,1,0]
data_frame = pd.DataFrame(x, columns=["x1","x2","x3"])
data_frame["y"] = y
# One hot jeden neuron zwraca jedną liczbę n klas n neuronów
data =multiply(data_frame,100)
y = data.loc[:,'y']
x = data.iloc[:,:-1]


norma = Normalization()
x = norma.normalizacja(x)
y= np.array(y)
x = np.array(x)

perc = Perceptron_functions(len_data=3,wyjscie_ilosc=1,epoki=100,activation="sigmoid")
perc.random_weights()
perc.random_bias()

perc.train_perceptron(x,y)
perc.perceptron(x,perc.wagi)
perc.activation = 'sigmoid'
perc.train_perceptron_back_propagation(x,y)
print('\n')

