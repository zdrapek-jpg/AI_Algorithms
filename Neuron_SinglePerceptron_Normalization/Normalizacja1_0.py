import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
class Normalization:
    # plan jest zeby przechowywać dla kolumn 1,2,3,4, itd po koleji elementy
    # przy inicjacji policzyć jedna daną zawirającą   wszystkie kolumny czyli [x1,x2,x3,x4...]
    # i znormalizować jeden obiekt  alby parametry były odpowiednie do trenowanych
    def __init__(self,data=None):
        if not  isinstance(data,pd.DataFrame) and data!=None:
            raise "dane muszą być jako dataframe"
        self.data= data
        self.minimum = []
        self.maximum = []
    # data ma kilka wymiarów i jest podany jak pandas frame
    def normalizacja(self,data):
        self.minimum=data.min().tolist()
        self.maximum= data.max().tolist()
        i=0
        new_data = pd.DataFrame()
        for MIN,MAX in zip(self.minimum,self.maximum):
            lista_el = data.iloc[:,i].values.tolist()
            lista =[round((x-MIN)/(MAX-MIN),4)        for x in  lista_el]

            new_data[f"x{i}"]=lista
            i+=1
        return new_data
    def normalize_point(self,point):
        if not (isinstance(point,list)):
            raise "Dane muszą być jako lista"
        return [round((point[i]-MIN)/(MAX-MIN),4)  for  i,(MIN,MAX) in enumerate(zip(self.minimum,self.maximum))]

import random

def randomiz(num):
    return (num + (random.random()*num)*0.5)


def multiply(datax, x):
    y1 = len(datax)
    new = []
    for _ in range(x+1):
        for _,row in datax.iterrows():

                new_ = [round(randomiz(row[0]), 2),round(randomiz(row[1]), 2),round(randomiz(row[2]), 2),row[3]]
                new.append(new_)

    new = pd.DataFrame(new, columns=datax.columns)
    data_x = pd.concat([new, datax], ignore_index=True)
    return data_x


def one_hot_encoder(x):
    """
    x is a sequence a list of object,
    x ->  take unique elements and create sqared matrix of data to get one hot enocder

    One hot  zwraca  liczbę n klas n neuronów    dla 3 klas 3 listy
    #1   [1,0,0]
    #2   [0,1,0]
    #3   [0,0,1]
    """
    x = [[el]  for el in set(x) ]
    df = pd.Series(x)
    print(df)
    MLB = MultiLabelBinarizer()
    response = pd.DataFrame(MLB.fit_transform(df), columns=MLB.classes_, index=df.index)
    return response

import numpy as np

def one_hot_transformation(labels):
    """Encodes a list of class labels into one-hot vectors."""
    classes = max(labels) + 1  # Number of unique classes
    encoded = np.zeros((len(labels), classes))
    for idx, val in enumerate(labels):
        encoded[idx, val] = 1
    return encoded

# Labels to encode
labels = [0, 1, 0, 0, 0, 1, 1, 1, 2]
encoder = one_hot_encoder(labels)
outputs_ = np.array([4.4,0.5,2.43])
def softmax(x):
    suma = sum(x)
    x_ = [out /suma  for out in x]
    forward = x_.index(max(x_))
    return encoder.index[forward]



# data_irys = pd.read_csv("iris_orig.csv", names=["x1", "x2", "x3", "x4", "label"])
# # wybranie  trzeciej i czwartej kolumny na dane do modelu
# cords_ = data_irys.iloc[:99, 1:]
# # dopisanie kolumny z wartośćią y [0,1] dla 2 kwiatków z przedział€ danych 0-90 wierszy
# cords_['label'], _ = pd.factorize(cords_["label"])
# x = cords_.iloc[:, 0:-1]
# obj =Normalization()
# x_normalized = obj.normalizacja(x)
# print(x_normalized.head())
# print(obj.minimum, obj.maximum)
#
