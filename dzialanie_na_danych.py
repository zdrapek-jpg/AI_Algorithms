import random

import K_means
import Tests_Kmeans

import string

Etykiety = [x for x in string.ascii_uppercase]

Data1 = [ ["cena",'zniżka','awokado','majonez'],
 [1, 13, 20, 41],
 [2, 12, 19, 72],
 [3, 12, 21, 7],
 [4, 12, 20, 37],
 [5, 10, 18, 8],
 [6, 10, 21, 22],
 [7, -11, 4, 84],
 [8, 1, 23, 9],
 [9, 5, 18, 63],
 [10, -5, 10, 13],
 [11, 0, 15, 11],
 [12, 11.3, 13.2, 66],
 [13, 22, 29, 22],
 [14, 25, 21, 20],
 [15, 20, 30, 59],
 [16, 24, 20, 61],
 [17, 20, 32, 37],
 [18, 20, 19, 13],
 [19, 16, 23, 83],
 [20, 16, 24, 89],
 [21, 14, 29, 63],
 [22, 17, 29, 17],
 [23, 15, 28, 9],
 [24, 16, 27, 6],
 [25, 16, 22, 65],
 [26, 12, 22, 75],
 [27, 18, 32, 7],
 [28, 14, 23, 39],
 [29, 20, 17, 9],
 [30, 16, 18, 25],
 [31, 15, 21, 23]]
Data2 = [["grozność",'wielkość'],
        [17, 13],
        [21, 25],
        [3, 12],
        [14.3, 6.9],
        [5, 10],
        [6, 10],
        [7, 11], [8, 1],
        [9, 5],
        [10, 5],
        [11, 0],
        [12, 11],
        [13, 22],
        [14, 25],
        [15, 20],
        [16, 24],
        [17, 20],
        [18, 20],
        [19, 16],
        [20, 16],
        [21, 14],
        [22, 17],
        [23, 15],
        [24, 16],
        [25, 16],
        [26, 12],
        [27, 18],
        [28, 14],
        [29, 20],
        [30, 16],
        [31, 15],
        [30,30],
        [19,21],
         [20,22],
         [21,19],
         [12,12],
         [10.5,32],
         [16.2,14.2],
         [10.2,14.7]
]


# # przekład na prostych danych  wpisywanych ręcznie
# Tests_main = Tests_Kmeans.Tests(Data2[1:],Etykiety,4)
# Tests_main.data_errors()
# Tests_main.labels_K_errors()
# K_means.initilaize_Kmenas(Data2[1:], 4, Etykiety)
#
#
# # przykładowe dane dla współrzędnych geograficznych danych rozrzuconych sklepów
# import pandas as pd
# data = pd.read_csv("Airbnb listings in Ottawa (May 2016).csv")
# data_cords = data.loc[: , ['latitude','longitude']]
# coords_list = data_cords.values.tolist()
# #
# K_means.initilaize_Kmenas(coords_list[:], 3, Etykiety=["A", "B", "C", "D", "E"])
# print(random.sample(Data2,1))


# # przekład na dnaych irysów na danych 2 wymiarowych po to aby była możliwość narysowania wykresów
import pandas as pd
data_irys = pd.read_csv("iris_orig.csv")
cords_ = data_irys.iloc[:,0:2]
cords_ = cords_.values.tolist()
K_means.initilaize_Kmenas(cords_,3,Etykiety,h=0.01)


