
data = [234,212,234,22,11,2334,5454,334,2231,233,555,444,222,551]
#normalizacja danych za pomocą skalowania standaryzacji z
 #s= odchylenie standardowe suma z (xi-xs)....
 # xs = średnia w zbiorze
 #x = wartość do normalizacji
from math import pow,sqrt

def odchylenie_standardowe(srednia,dane):
    suma = sum([pow((x-srednia),2 ) for x in dane])

    return round(sqrt(suma/len(dane)),4)

def standaryzacja_data(data,flag = True):
    suma =  sum(data)
    dl = len(data)
    xs = suma/dl
    s =odchylenie_standardowe(xs,data)

    normalized=   [round(((x-xs)/s),4)  for x in data ]
    print(f" po normalizacji {normalized}, {s}")
    dane = [ x for x,y in zip (data,normalized) if y >-2.5 and y<2.5]
    print(f"dane pomnijeszod po odcięci punktów granicznych {data} obicęto {len(data)-len(dane)} :")
    if flag:
        return standaryzacja_data(dane,flag = False)
    return normalized


print(standaryzacja_data(data))

