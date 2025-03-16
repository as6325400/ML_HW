import numpy as np


class DataLoader():

    def __init__(self, path: str) -> None:
        """
        The meaning of varibles:
        data -> list, n * 3 matrix, n = nums of data points
        label -> list, n * 1, label of coordinate point(x,y)

        ### These four variables are used to draw the data points ###
        cor_x_pos -> list, (n/2) * 1,coordinate point of x axis with postive label 
        cor_y_pos -> list, (n/2) * 1,coordinate point of y axis with postive label
        cor_x_neg -> list, (n/2) * 1,coordinate point of x axis with negative label
        cor_y_neg -> list, (n/2) * 1,coordinate point of y axis with negative label

        """
        self.data, self.label, self.cor_x_pos, self.cor_y_pos, self.cor_x_neg, self.cor_y_neg = self.read_data(
            path)

    def read_data(self, path: str) -> list:
        """
        To read data coordinate points and put them into 'list' data type

        """
        with open(path, 'r') as raw_data:
            lines = raw_data.readlines()
            data = []       # dataset
            label = []      # label
            # These four variables are used to draw the data points
            cor_x_pos = []  # data with +1 label in x axis
            cor_y_pos = []  # data with +1 label in y axis
            cor_x_neg = []  # data with -1 label in x axis
            cor_y_neg = []  # data with -1 label in y axis

            for words in lines:
                words = words.strip().split(' ')
                for idx in range(4):
                    words[idx] = float(words[idx])  # type: ignore
                    if idx == 1:
                        if int(words[3]) == 1:
                            cor_x_pos.append(words[idx])
                        elif int(words[3]) == -1:
                            cor_x_neg.append(words[idx])
                    if idx == 2:
                        if int(words[3]) == 1:
                            cor_y_pos.append(words[idx])
                        elif int(words[3]) == -1:
                            cor_y_neg.append(words[idx])

                data.append(words[0:3])
                label.append(words[3])

            return data, label, cor_x_pos, cor_y_pos, cor_x_neg, cor_y_neg  # type: ignore
