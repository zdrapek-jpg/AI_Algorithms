import logging
from typing import List, Any
logging.basicConfig(
    filename="ProgramLogs.txt",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
with open("ProgramLogs.txt", "w") as file:
    file.write(" ")


class Tests():
    errors_list = []

    def __init__(self, data: List[Any], labels_points: List[str], k: int, i: int = 0):
        self.Data= data
        self.labels_points = labels_points
        self.k=k
        self.i=i
        logging.info(f"Initialized Tests class. Object ID: {self.i}")


    # gdy dane nie są listą
    # gdy dane są krótsze o 2 nie ma sensu uruchamiać algorytmu
    # dane przekazane muszą być listą
    # jeśli jedna z zmiennyh   jest typu innego niż  int lub float
    # gdy jeden z punktów ma inną długość danych niż pozostałe
    def data_errors(self):
        try:
            if len(self.Data) <= 2:
                logging.error("Data is too short: length %d     %d", len(self.Data),self.i)
                raise TestKmeans_Info(f"Data provided for model should be longer than 2 and is: {len(self.Data)}")

            if not isinstance(self.Data, list):
                logging.error("Data type is not a list: %s  ID: %d", type(self.Data), self.i)
                raise TestKmeans_data(f"Data provided for model must be type of list, not {type(self.Data)}")

            x_basic_length = len(self.Data[0])
            for one_line in self.Data:
                if not isinstance(one_line, list):
                    logging.error("Element type is not a list: %s  ID: %d", type(one_line), self.i)
                    raise TestKmeans_data(
                        "Object type error",
                        f"Object type supposed to be a list, not {type(one_line)}",
                        f"{self.Data.index(one_line)}"
                    )

                if x_basic_length != len(one_line):
                    logging.error("Inconsistent data lengths: %d != %d  ID: %d", len(one_line), x_basic_length, self.i)
                    raise TestKmeans_data(
                        "Different length",
                        f"Objects with {len(one_line)} != {x_basic_length}",
                        f"{self.Data.index(one_line)} and 0"
                    )

                if not all(isinstance(element, (int, float)) for element in one_line):
                    logging.error("Invalid element types in data: %s   ID: %d", one_line, self.i)
                    raise TestKmeans_data(
                        "Wrong type",
                        f"Error occurred for: {one_line}, all numbers must be int or float",
                        self.Data.index(one_line)
                    )
            logging.info("Data passed  ID: %d", self.i)
        except Exception as e:
            logging.exception("Exception in data_errors.")
            raise

    #gdy etykiety są  rużnych typów wyrzucamy błąd
    # gdy k jest większe od ilości naych lub k jest większe od ilości etykiet
    # gdy k jest większe od ilości etykiet
    def labels_K_errors(self):
        try:
            if not isinstance(self.k, int):
                logging.error("k is not an integer: %s   ID: %d", type(self.k), self.i)
                raise TestKmeans_data("k is wrong type", f"{type(self.k)} must be <class 'int'>", "0")

            if self.k > len(self.Data) or self.k <= 0:
                logging.error("Invalid k value: k=%d, data length=%d  ID: %d", self.k, len(self.Data), self.i)
                raise TestKmeans_Info(
                    f"""k count of clusters must be greater than 0 and lower or equal to count of data.
                    k={self.k}, data length={len(self.Data)}"""
                )

            if self.k > len(self.labels_points):
                logging.error("k is larger than the number of labels: k=%d, labels=%d   ID: %d",
                              self.k,
                              len(self.labels_points),
                              self.i)
                raise TestKmeans_area(
                    "k count of clusters must be less than or equal to the labels you provide",
                    f"{self.k},{len(self.labels_points)}"
                )
            if self.k > len(self.Data) or len(self.labels_points)>len(self.Data):
                logging.error("k is larger  or equal to number of labels and less than data: k=%d, labels=%d data=%d   ID: %d",
                              self.k,
                              len(self.labels_points),
                              len(self.Data),
                              self.i)
                raise TestKmeans_area(
                    "k count of clusters must be less than or equal to the labels you provide",
                    f"{self.k},{len(self.labels_points)}"
                )

            if not all(isinstance(element, str) for element in self.labels_points):
                logging.error("Labels contain non-string elements: %s  ID: %d", self.labels_points, self.i)
                raise TestKmeans_area(
                    "Center Points Data invalid",
                    f"Elements must be strings, found: {self.labels_points}"
                )
            logging.info("Labels and k passed all checks. ID: %d", self.i)
        except Exception as e:
            logging.exception("Exception in labels_K_errors for Object ID: %d.", self.i)
            raise

    @classmethod
    def continuation_conditions_centers(cls,current_centers,previous_centers):
        # na aktualnych centrach możęmy sprawdzić czy centra mają odwpiednią długość czy nie brakuje danych lub jest ich za dużo
        sprawdzenie = {}
        x = current_centers[0]
        sprawdzenie = [x]
        for y in current_centers[1:]:
            if len(x) != len(y):
                logging.warning("Centers have inconsistent lengths.")
                return False
            sprawdzenie.append(y)

        if len(current_centers) != len(sprawdzenie):
            return False
        logging.warning("Mismatch in center lengths.")
        if len(previous_centers) != len(current_centers):
            logging.warning("Previous and current centers have different lengths.")
            return False  # Grupy mają różną liczbę elementów kontynuujemy

        # current_centers i previous_centers powinny być listami o tej samej strukturze
        total_error = 0
        count = 0
        same= 1
        for prev_group, curr_group in zip(previous_centers, current_centers):
            if prev_group == curr_group:
                same+=1
            for c, p in zip(prev_group[:-1], curr_group[:-1]):
                total_error += abs(c - p)
                count += 1
        # Obliczamy średni błąd bezwzględny
        mean_absolute_error = total_error / count if count != 0 else 0
        print("error",mean_absolute_error)
        logging.debug("Mean absolute error: %f", mean_absolute_error)
        # sprawdzamy czy jest więszky od warotści
        print(f"{mean_absolute_error} < {0.50} and {True} and {same/len(current_centers)}>=1")
        logging.debug(f" conditons result: {mean_absolute_error} < {0.50} and {True} and {same/len(current_centers)}>=1")
        return (mean_absolute_error < 0.39 and True) and same/len(current_centers)>=1   #kontynuujemy jeśli błąd jest mniejszy od 0.39 albo żadne z powyższych nie zwruciły inaczej
        #
        #([[1, 13], [2, 12], [3, 12], [4, 12], [5, 10], [6, 10], [7, -11], [8, 1], [9, 5], [10, -5], [11, 0], [12, 11.3],
        #   [13, 22], [14, 25], [15, 20], [16, 24], [17, 20], [18, 20], [19, 16], [20, 16], [21, 14], [22, 17], [23, 15],
        #   [24, 16], [25, 16], [26, 12], [27, 18], [28, 14], [29, 20], [30, 16], [31, 15]]
    @classmethod
    def continuation_conditions_groups(cls,previous_group,current_group):
        try:
            for p, c in zip(previous_group, current_group):
                for points_p, points_c in zip(p, c):
                    if points_c != points_p:
                        logging.debug("Groups differ: %s != %s", points_p, points_c)
                        return False
                    for pp, cc in zip(points_p, points_c):
                        if pp != cc:
                            logging.warning("Point elements differ: %s != %s", pp, cc)
                            return False
            logging.debug("Groups are stable.")
            return True
        except Exception as e:
            logging.exception("Exception in continuation_conditions_groups.",e)
            raise

class TestKmeans_Info(Exception):
    pass
    def __init__(self,text):

        super().__init__(text)
    def __str__(self):
        return '\033[93m'+"{} ".format(super().__str__())


class TestKmeans_area(TestKmeans_Info):
    def __init__(self,text,area):
        super().__init__(text)
        self.area= area

    def __str__(self):
        return "{} at: {} in data".format(super().__str__(),self.area)
class TestKmeans_data(TestKmeans_Info):

    def __init__(self,text,data,index):
        super().__init__(text)
        self.data=data
        self.index=index

    def __str__(self):
        return "{} with: \033[91m{} , \n at index : {}".format(super().__str__(),self.data,self.index)


