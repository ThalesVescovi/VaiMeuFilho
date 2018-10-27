import numpy as np
import cv2
import algorithms.lbp as lbp
import sys
import ast
import algorithms.lbp as lbp

def main():
    # Feature set containing (x,y) values of 25 known/training data
    trainData = np.random.randint(0, 100, (50, 5)).astype(np.float32)

    # Labels each one either Red or Blue with numbers 0 and 1
    responses = np.random.randint(0, 5, (50, 1)).astype(np.float32)

    print(trainData)
    print(responses)

    knn = cv2.ml.KNearest_create()
    knn.train(trainData, cv2.ml.ROW_SAMPLE, responses)

    print(np.random.randint(0, 100, (1, 5)))
    newcomer = np.random.randint(0, 100, (1, 5)).astype(np.float32)
    print(newcomer)

    ret, results, neighbours, dist = knn.findNearest(newcomer, 3)

    print("ret: ", ret, "\n")
    print("result: ", results, "\n")
    print("neighbours: ", neighbours, "\n")
    print("distance: ", dist)

if __name__ == '__main__':
	main()