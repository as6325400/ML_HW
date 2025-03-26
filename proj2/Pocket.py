import numpy as np
import argparse
from DataLoader import DataLoader
import os
import matplotlib.pyplot as plt
import time

def count_error_num(weight_matrix: np.array, DataLoader: DataLoader) -> int:
    data_points = np.array([[1, d[1], d[2]] for d in DataLoader.data])
    predictions = np.sign(np.dot(data_points, weight_matrix))
    return sum(predictions != DataLoader.label)
    

def pocket(DataLoader: DataLoader) -> np.ndarray:
    """
    Do the Pocket algorithm here on your own.
    weight_matrix -> 3 * 1 resulted weight matrix  

    """
    weight_matrix = np.zeros(3)
    s = time.time()
    ############ START ##########
    max_iterations = 50
    error_num = count_error_num(weight_matrix, DataLoader)
    indices = np.random.permutation(len(DataLoader.data))
    DataLoader.data = [DataLoader.data[i] for i in indices]
    DataLoader.label = [DataLoader.label[i] for i in indices]
    
    
    for _ in range(max_iterations):
        if error_num == 0:
            break
        
        for idx in range(len(DataLoader.data)):
            point = np.array([1, DataLoader.data[idx][1], DataLoader.data[idx][2]])
            if np.sign(np.dot(weight_matrix, point)) != DataLoader.label[idx]:
                tmp_weight_matrix = weight_matrix + DataLoader.label[idx] * point
                tmp_error_num = count_error_num(tmp_weight_matrix, DataLoader)
                if  tmp_error_num < error_num:
                    weight_matrix = tmp_weight_matrix
                    error_num = tmp_error_num       
    ############ END ############
    e = time.time()
    print("ex time = %f" % (e-s))
    print(f"error nums: {count_error_num(weight_matrix, DataLoader)}")
    return weight_matrix


def main(args):
    try:
        if args.path == None or not os.path.exists(args.path):
            raise
    except:
        print("File not found, please try again")
        exit()

    Loader = DataLoader(args.path)
    updated_weight = pocket(DataLoader=Loader)

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
