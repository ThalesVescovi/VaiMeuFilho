import numpy as np
import cv2
import algorithms.cmct as cmct
import sys
from matplotlib import pyplot as plt
import psycopg2


def main():
    arqHistSujeitos = open("C:\\Users\\thale\\PycharmProjects\\VaiMeuFilho\\bd_projeto\\histograms\\base_histogramas_cmct.txt", "w")
    arqHistSujeitosTeste = open("C:\\Users\\thale\\PycharmProjects\\VaiMeuFilho\\bd_projeto\\histograms\\teste_histogramas_cmct.txt", "w")
    print('--------- CMCT ---------')
    print('EXTRAINDO CARACTERISTICAS')

    for sujeitoNum in range(1, 41):
        chaveSujeitoTeste = np.random.randint(1, 11)  # indice da chave usada para teste altera a cada sujeito lido
        for faceNum in range(1, 11):
            path = ('C:\\Users\\thale\\PycharmProjects\\VaiMeuFilho\\bd_projeto\\att_faces\\s' + str(sujeitoNum) + '\\' + str(faceNum) + '.pgm')

            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            imgLimiarizada = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            hist = cmct.cmct_function(img, imgLimiarizada)
            if (faceNum != chaveSujeitoTeste):
                arqHistSujeitos.write(str(sujeitoNum)+';'+str(faceNum)+';'+str(hist)+'\n')
            else:
                arqHistSujeitosTeste.write(str(sujeitoNum)+';'+str(faceNum)+';'+str(hist)+'\n')
        # end for faceNum
        print(str(sujeitoNum) + ' Done')
    # end for sujeitoNum
    print('SALVANDO HISTOGRAMAS')

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    arqHistSujeitos.close()
    arqHistSujeitosTeste.close()

    print('PRONTO!')



if __name__ == '__main__':
	main()