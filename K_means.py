#!Python 10  k-means algorithm
##        (X,X,X,X,X), (X,X,X,X,X,X) PUNKTY BEZ ETYKIET
     #            ETYKIETY PODAJE JAKO K : A,B,C itd..
     #      DO PIERWSZYCH PUNKTÓW O ILE  K> ILOSC PUNKTÓW
     #i k >1 inaczej przypisuje wszystko do 1 klasy
#       KROKI :
        #1   PRZYPISANIE ETYKIET DO 3 LOSOWYCH PUNKTÓW
        #2   PUNKTY BIORĘ JAKO STAŁE I LICZE DYSTANSE DO KAŻDEGO  PUNKTU Z TYM ŻE PKT SAM DO SIEBIE MA 0 DYSTANS
        #3    klasa 1 [0[] 1[],2[]] , klasa 2 [ 0[] 1[],2[]]  DYSTANSE,  biorę minimum
import random
# będzie jedna etykieta dla wszystkich punktów
# przypisanie etykiet do grup :
# jak zapisać etykietę  na krotce? -1 jest etykietą i lista obiektów jest jako lista klastrów poszczegulnych klas po kolei
# (x,x,x,klasa)
# punkty z odlgłościami  muszą być abstrakcyjne i tymczasowe  (opltymalizacja pętli liczenia dystansów)
# stworzenie funkcji  która definiuje kiedy powinniśmy przestać przesuwanie centroidów gdyby python miał problem z liczeniem
# pętlka rekuręcyjna która powtarza praces liczenia algorytmu

from math import sqrt
from matplotlib import pyplot as plt
import Tests_Kmeans
from Tests_Kmeans import Tests

# metryka licząca dystanse między centrum a każdym punktem w uniwersum
def sumy_odwrotnosci_metryka(added_point, distance_data): # x1(2,4,4,3)  A(4,6,5,6)
    if len(added_point)!= len(distance_data):
        raise  Exception(f"nieodpowiednie dystane dla punktów miedzy sobą {len(added_point)}, {len(distance_data)} ")
    suma = [(e1-e2)**2 for e1,e2 in zip(added_point,distance_data)]
    #return  Pierwiastek (suma (dla 2 listy między sobą))
    return round(sqrt(sum(suma)),4) # 5.322245

#przypisanie punktu etykiety centrum
def get_centers(one_data,Etykieta,i=0):  # (2,3)  "A"  - >   (2,3,"A")
    return list(one_data[0])+[Etykieta]
# po udanej pierwszej iteracji wyznaczania dystansów wyliczamy nowe centra z punktów przypisanych do poszczegulnych grup
def new_center_set(Data,etykiety):
    etykiety = [x[:][-1] for x in etykiety]
    i = 0
    centers = []
    for point in Data:
        assert ( isinstance(etykiety[i],str)),"the label for points must be str object"
        number_oof_points = len(point)
        suma = [sum(points_data)/number_oof_points for points_data in zip(*point)]
        center = suma+[etykiety[i]]
        centers.append(center)
        i+=1
    return centers

# wyszukanie blędó w algorytmie sprawdzenie o ile przesunęły się punkty + połączenie z inną funkcją czy punkty zmienił wartość sprawdza czy kontynuować iterację


# algorytm sprawdzający czy  kontynuujemy ze wzgłedy na centra

      # Grupy są stabilne, można zakończyć algorytm, nic się nie zmineiło
# check if point for labels are the same
# sprawdzamy czy punkty mają te same współrzędne co inny punkt zwracamy wtedy error  else: kontynuujemy
import numpy as np
import numpy as np
import matplotlib.pyplot as plt


def sumy_odwrotnosci_metryka_np(added_point, distance_data):
    # Ensure both added_point and distance_data are numeric
    added_point = np.array(added_point, dtype=float)
    distance_data = np.array(distance_data, dtype=float)

    # Calculate the squared differences and sum them
    suma = np.sum((added_point - distance_data) ** 2)
    return suma
def extra_Draw(Data, centers, groups, Etykiety, k,h):
    if h ==None:
        h = 0.1
    cords = np.array(Data)
    centers = np.array(centers)
    centroids = centers[:, :2].astype(float)

    x_min, x_max = cords[:, 0].min()-h , cords[:, 0].max()+h
    y_min, y_max = cords[:, 1].min()-h , cords[:, 1].max()+h
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Calculate decision boundary: assign each point in the grid to the nearest centroid
    Z = np.array([
        np.argmin([sumy_odwrotnosci_metryka_np(center[:-1], [x, y]) for center in centers])
        for x, y in np.c_[xx.ravel(), yy.ravel()]
    ])
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary
    plt.imshow(Z, interpolation='nearest', extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Pastel1, origin='lower')

    # Plot data points for each cluster
    for idx, group in enumerate(groups):
        points = np.array(group)
        plt.scatter(points[:, 0], points[:, 1],s=10,label =Etykiety[idx])
    plt.scatter(centroids[:, 0], centroids[:, 1], marker="*",s=10)
    plt.grid()
    plt.show()
# rysowanie punktów w przestrzeni 2 wymiarowej
def draw_with_etykiets_data_points(groups,centers):  #  data[[x,y....],[],[],[],], centers[[X,Y,...,A],[X,Y,....,B],[X,Y,....,C]]
    # [[1, 13], [2, 12], [3, 12], [4, 12], [5, 10], [6, 10], [7, -11], [8, 1], [9, 5], [10, -5], [11, 0], [12, 11.3],
    #   [13, 22], [14, 25], [15, 20], [16, 24], [17, 20], [18, 20], [19, 16], [20, 16], [21, 14], [22, 17], [23, 15],
    #   [24, 16], [25, 16], [26, 12], [27, 18], [28, 14], [29, 20], [30, 16], [31, 15]],
    #  [[1, 13, 'A'], [2, 12, 'B'], [17, 13, 'C']]
    if len(groups)!= len(centers):
        raise Exception(f"One of data is not long enought len(groups)!=len(centers) ,{len(groups)}!={len(centers)} ")
    fig,ax = plt.subplots()
    for group,center in zip(groups,centers):
        if not isinstance(group, (int, float)) and not all(isinstance(x, (int, float)) for x in center[:-1]):
            raise Exception("One of data is incorrect type (string or other")
        x = [point[0] for point in group]
        y = [point[1] for point in group]
        plt.scatter(x, y,  label=center[-1], s=20)
        for i in range(len(x)):
            ax.annotate(center[-1], (x[i], y[i]), textcoords="offset points", xytext=(0, 4), ha='center')

        plt.scatter(center[0],center[1],marker="*",label= center[-1],s=10)
    plt.grid()
    plt.show()

heads = 0
def initilaize_Kmenas(Data,k,Etykiety,i=0,h=None,cluster_center=None,groups_before= None):
    global heads
    if  k==1:
        return   [x+[Etykiety[i]] for x in Data[1:]]

    else:
        Data_etyk = []
        groups= []
        if cluster_center==None :
            cluster_center=[]
            for index in range(k):
                # Calculate center for each cluster and add to Data_etyk
                cluster_center = get_centers(random.sample(Data,1), Etykiety[index])
                Data_etyk.append(cluster_center)
                groups.append([])

        else:
            Data_etyk = [x for x in cluster_center]
            groups = [[] for x in cluster_center]
        # for index  in range(k):
        #     # Calculate center for each cluster and add to Data_etyk
        #     cluster_center = get_centers(Data[index], Etykiety[index])
        #     Data_etyk.append(cluster_center)
        #     groups.append([])

        for tp in Data:
            distance_for_current_center = []
            for center in Data_etyk:
                distance_for_current_center.append(sumy_odwrotnosci_metryka(center[:-1], tp))

            min_distance_point = min(distance_for_current_center)
            index = distance_for_current_center.index(min_distance_point)
            groups[index].append(tp)

    draw_with_etykiets_data_points(groups,Data_etyk)
    new_centers = new_center_set(groups,Data_etyk)

    heads += 1
    flag1 = Tests.continuation_conditions_centers(Data_etyk, new_centers) #if true wszystko spełnione zatrzymujemy algorytm
    flag2= False
    print(f"flag for centers:{flag1} ")
    if heads >=2:
         flag2 = Tests.continuation_conditions_groups(groups_before, groups) # if true zatrzymujmey
         print(f"flag for groups: {flag1}")
    print(f"before : {Data_etyk} after:{new_centers}")
    ### petla do poprawy
    # for i in range(len(Data_etyk)):
    #     if not flag2 and not flag1:
    #        if heads <=16:
    #             return initilaize_Kmenas(Data, k,Etykiety, cluster_center=new_centers,groups_before=groups)
    if not flag2 and  not flag1 and heads <= 20:
        return initilaize_Kmenas(Data, k, Etykiety,h=h, cluster_center=new_centers, groups_before=groups)
    extra_Draw(Data,new_centers,groups,Etykiety,k,h)
    return Data,new_centers,groups,Etykiety,k

