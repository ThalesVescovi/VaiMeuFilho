import numpy as np
import cv2
import algorithms.cmct as cmct
import sys
from matplotlib import pyplot as plt
import psycopg2


def main():
    arqHistogramas = open("C:\\Users\\thale\\PycharmProjects\\VaiMeuFilho\\bd_projeto\\att_faces_histogramas\\histogramas_cmct.txt.txt", "w")
    print('--------- CMCT ---------')
    print('EXTRAINDO CARACTERISTICAS')

    for sujeitoNum in range(1, 41):
        for faceNum in range(1, 11):
            path = ('C:\\Users\\thale\\PycharmProjects\\VaiMeuFilho\\bd_projeto\\att_faces\\s' + str(sujeitoNum) + '\\' + str(faceNum) + '.png')
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            hist = cmct.cmct_function(img)
            arqHistogramas.write(str(sujeitoNum)+';'+str(faceNum)+';'+str(hist)+'\n')
        # end for faceNum
        print(str(sujeitoNum) + ' Done')
    # end for sujeitoNum
    print('SALVANDO HISTOGRAMAS CMCT')

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    arqHistogramas.close()

    print('PRONTO!')



if __name__ == '__main__':
	main()