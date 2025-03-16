import numpy as np
import argparse
import os
from DataLoader import DataLoader
import matplotlib.pyplot as plt
import time


def PLA(DataLoader: DataLoader) -> np.ndarray:
    """
    Do the PLA here on your own.
    weight_matrix -> 3 * 1 resulted weight matrix  

    """
    weight_matrix = np.zeros(3)
    s = time.time()
    ############ START ##########

    ############ END ############
    e = time.time()
    print("ex time : %f" % (e-s))
    return weight_matrix


def main(args):
    try:
        if args.path == None or not os.path.exists(args.path):
            raise
    except:
        print("File not found, please try again")
        exit()

    Loader = DataLoader(args.path)
    updated_weight = PLA(DataLoader=Loader)

    # This part is for plotting the graph
    plt.xlim(-1000, 1000)
    plt.ylim(-1000, 1000)
    plt.scatter(Loader.cor_x_pos, Loader.cor_y_pos,
                c='b', label='pos data')
    plt.scatter(Loader.cor_x_neg, Loader.cor_y_neg,
                c='r', label='neg data')

    x = np.linspace(-1000, 1000, 100)
    # This is the base line
    y1 = 3*x+5
    # This is your split line
    y2 = (updated_weight[1]*x + updated_weight[0]) / (-updated_weight[2])
    plt.plot(x, y1, 'g', label='base line', linewidth='1')
    plt.plot(x, y2, 'y', label='split line', linewidth='1')
    plt.legend()
    plt.show()


if __name__ == '__main__':

    parse = argparse.ArgumentParser(
        description='Place the .txt file as your path input')
    parse.add_argument('--path', type=str, help='Your file path')
    args = parse.parse_args()
    main(args)
