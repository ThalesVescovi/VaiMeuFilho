import numpy as np
import cv2
import algorithms.mct8 as mct8
import sys
from matplotlib import pyplot as plt
import psycopg2


def main():
    arqHistogramas = open("C:\\Users\\thale\\PycharmProjects\\VaiMeuFilho\\bd_projeto\\att_faces_histogramas\\histogramas_mct8.txt", "w")

    print('--------- MCT8 ---------')
    print('EXTRAINDO CARACTERISTICAS')

    for sujeitoNum in range(1, 41):
        for faceNum in range(1, 11):
            path = ('C:\\Users\\thale\\PycharmProjects\\VaiMeuFilho\\bd_projeto\\att_faces\\s' + str(sujeitoNum) + '\\' + str(faceNum) + '.png')
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            hist = mct8.mct8_function(img)
            arqHistogramas.write(str(sujeitoNum)+';'+str(faceNum)+';'+str(hist)+'\n')
        # end for faceNum
        print(str(sujeitoNum) + ' Done')
    # end for sujeitoNum
    print('SALVANDO HISTOGRAMAS MCT8')

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    arqHistogramas.close()

    print('PRONTO!')



if __name__ == '__main__':
	main()