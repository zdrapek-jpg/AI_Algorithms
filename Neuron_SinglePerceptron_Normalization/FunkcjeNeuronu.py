import pandas as pd
import numpy as np
import random
import time
from Normalizacja1_0 import multiply, Normalization

class Perceptron_functions:
    def __init__(self, len_data, wyjscie_ilosc=1,epoki=None,bias=None,activation=None):
        self.wyjscia_ilosc = wyjscie_ilosc
        self.len_data= len_data
        self.activation=activation
        self.wagi=None
        self.prog=0.6
        self.alfa=0.15
        self.bias =bias
        self.epoki = epoki
        if epoki is None:
            self.epoki = 2
        self.one_hot_encoder = None
        self.loss=[]
    def forward(self,x):
        pass
    def train_perceptron(self, x_train, y_train):
        for j in range(1, self.epoki + 1):
            err=0
            for i,point in enumerate(x_train):
                sumy_wazone = np.dot(point,self.wagi)
                y_pred = self.aktywacja(sumy_wazone)
                blod_e = y_train[i] - y_pred
                delta_wag = self.alfa * point * blod_e
                self.wagi= self.wagi + delta_wag
                err+=(0.5 * (1 - y_pred) ** 2)
            self.loss.append(err/len(y_train))
            self.perceptron(x_train, self.wagi)
            outcome = list(map(lambda x: round(x), self.output))
            acc = sum([1 if x == y else 0 for x, y in zip(outcome, y_train)]) / len(y_train)
            print("loss: ", round(self.loss[-1],6), end="  ")
            print("accuracy: ", round(acc,3))
            time.sleep(0.1)
            if acc > 0.99 and self.loss[-1]<0.02:
                return True, acc,self.loss[-1],i
        return True, acc,self.loss[-1],i



    def train_perceptron_back_propagation(self,x_train,y_train):
        for i in range(1, self.epoki + 1):
            err = 0
            for j,point in enumerate(x_train):
                suma_wazona = np.dot(point,self.wagi)+self.bias
                y_pred = self.aktywacja(suma_wazona)
                err+=(0.5*(y_train[j]-y_pred)**2)
                pochodna_wyjscia = -(y_train[j]-y_pred)
                pochodna_funkcji = y_pred*(1-y_pred)
                pochodna  = point * pochodna_wyjscia*pochodna_funkcji
                self.wagi = self.wagi - (np.full(3,self.alfa)*np.full(3,pochodna))
            self.loss.append(err/len(y_train))
            self.perceptron(x_train, self.wagi)
            outcome = list(map(lambda x: round(x), self.output))
            acc = sum([1 if x == y else 0 for x, y in zip(outcome, y_train)]) / len(y_train)
            print("loss: ",round(self.loss[-1],6),end="  ")
            print("accuracy: ", round(acc,3))
            time.sleep(0.1)
            if acc > 0.991 and self.loss[-1]<0.022:
                print("epoka:",i)
                return True, acc,self.loss[-1],i
        return True, acc,self.loss[-1],i


    def perceptron(self,wejscia, wagi):
        self.output= []
        for i,point in enumerate(wejscia):
            suma_wazona = np.dot(point,wagi)+self.bias
            #print(suma_wazona,end=" ")
            self.output.append(self.aktywacja(suma_wazona).tolist())

    def aktywacja(self,x):
        if self.activation=="prog":
            return x >= self.prog, 1, 0
        if self.activation=="sigmoid":
                return 1 / (1 + np.exp(-np.array(x)))
        else:
            raise "nie wybrano żadnej funkcji aktywacji"

    def activation_soft_max(self,x:list) ->list:  # x
        suma = sum(x)
        x_ = [out / suma for out in x]
        forward = x_.index(max(x_))
        return self.one_hot_encoder.index[forward]

    def random_weights(self):
        self.wagi =np.array([round(random.random() * random.choice([1,1,0.8,0.8]),3)  for i in range(0,self.len_data)] )
# {w1,w2,w3}{w1,w2,w3}
    def random_bias(self):
        self.bias =np.array(round(random.random() * random.choice([0.9,0.9,0.8,0.8]),3))

    def random_alfa(self):
        self.alfa=  round(random.random()*0.7,3)

    def random_prog(self):
        self.prog =  np.array(round(random.random()*1.1,3))

