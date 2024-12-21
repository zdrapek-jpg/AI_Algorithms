from  FunkcjeNeuronu import Perceptron_functions
from Normalizacja1_0 import Normalization, multiply
import math
import pandas as pd


class Ada_Boost:
    def __init__(self, x_train, y_train, iterations):
        self.alfa = []
        self.classifiers = []
        self.x_train = x_train
        self.y_train = y_train
        self.iterations = iterations
        self.sample_weights = []

    def fit(self):
        # Initialize sample weights
        self.sample_weights = [1 / len(self.y_train)] * len(self.y_train)

        for i in range(self.iterations):
            perc = Perceptron_functions(3, epoki=2, activation="sigmoid")
            perc.train(self.x_train, self.y_train, self.x_train, self.y_train, proby=1)
            perc.perceptron(self.x_train, perc.wagi)  # Get predictions
            y_pred = list(map(lambda x: round(x), perc.output))

            # Calculate weighted error
            weighted_error = sum(
                self.sample_weights[i] if y1 != y2 else 0
                for i, (y1, y2) in enumerate(zip(self.y_train, y_pred))
            )
            print(f"Weighted error: {weighted_error}")

            # If error is 0, the weak classifier is perfect
            if weighted_error == 0:
                self.alfa.append(float('inf'))
                self.classifiers.append(perc)
                break

            if weighted_error >= 0.5:
                print("Classifier is no better than random. Skipping iteration.")
                break
            alpha = 0.5 * math.log((1 - weighted_error) / weighted_error)

            self.sample_weights = [
                w * math.exp(-alpha) if y_true == y_pred else w * math.exp(alpha)
                for w, y_true, y_pred in zip(self.sample_weights, self.y_train, y_pred)
            ]


            weight_sum = sum(self.sample_weights)
            self.sample_weights = [w / weight_sum for w in self.sample_weights]
            print(f"Updated weights: {self.sample_weights},alpha {self.alfa}")
            self.alfa.append(alpha)
            self.classifiers.append(perc)

    def predict(self, X):
        final_prediction = [0] * len(X)  # Initialize with zeros

        for alpha, classifier in zip(self.alfa, self.classifiers):
            # Get predictions from the classifier
            classifier.perceptron(X, classifier.wagi)  # Use perceptron to get predictions
            y_pred = list(map(lambda x: round(x), classifier.output))  # Round to get 0 or 1 predictions

            # Update final predictions by adding weighted predictions
            final_prediction = [
                alpha * cls_pred for  cls_pred in  y_pred
            ]

        # Final prediction: sign of the aggregated score (threshold at 0)
        return [1 if y >0 else 0 for y in final_prediction]

datax = [[2.5, 3.1, 10, 1],[8, 6, 7, 0],[11, 7, 4, 0],[11.2, 8, 4, 0],[3, 1.5, 11, 1],[0.1, 0.6, 13, 1],[2.6, 2.1, 8, 1]
]
0.5

0.56
alph =0.8
acc= 1.0
#cel wykrzystanie gradientu
# funkcja straty mozę być dowolna dla gorszych wynikow  funkcja straty ma minima lokalne
# algorytm znajdzie minimum lokalne adaboost gdy  znjdzie zbyt mały bląd minlokalne cofa się i szuka
#lepszego punktu do zopymalizowania
#algorytm adam
# soft max , # gxboost


datax = [[2.5, 3.1, 10], [8, 6, 7], [11, 7, 4], [11.2, 8, 4], [3, 1.5, 11], [0.1, 0.6, 13], [2.6, 2.1, 8]]
# zapisanie tablicy w dataframe
data_frame = pd.DataFrame(datax, columns=['X1', 'X2', 'X3'])
y = [1, 0, 0, 0, 1, 1, 1]
data_frame["y"] = y
x =multiply(data_frame,4)
print(x.head, y)
# normalizacaj  danych  wejściowych
obj = Normalization()

normalized_frame = obj.normalizacja(data_frame)

adaboost = Ada_Boost(normalized_frame, y, iterations=6)
adaboost.fit()
predictions = adaboost.predict(normalized_frame)
print("wyniki:")
perc = Perceptron_functions(len_data=3,epoki=3,activation="sigmoid")
perc.output = y
print(perc.check_accuracy(predictions))

perc.train(normalized_frame,y,normalized_frame,y,1)



