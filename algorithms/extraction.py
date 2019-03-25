import numpy as np
import cv2
import algorithms.lbp as lbp
import sys
from matplotlib import pyplot as plt
import psycopg2

from algorithms import mct8, cmct


def main():
    # PARAMETROS PARA LEITURA DOS ARQUIVOS
    #nomeBD = "att_faces"; qtdS = 40; lstQtdFaces = [10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,]
    #nomeBD = "umist"; qtdS = 20; lstQtdFaces = [38,35,26,24,26,23,19,22,20,32,34,34,26,30,19,26,26,33,48,34]
    nomeBD = "jaffe"; qtdS = 10; lstQtdFaces = [23,22,22,20,21,21,20,21,21,22]

    print('EXTRAINDO CARACTERISTICAS')

    arqHistogramasLBP = open("C:\\Users\\thale\\PycharmProjects\\VaiMeuFilho\\bd_projeto\\" + nomeBD + "_histogramas\\histogramas_lbp.txt", "w")
    arqHistogramasMCT8 = open("C:\\Users\\thale\\PycharmProjects\\VaiMeuFilho\\bd_projeto\\" + nomeBD + "_histogramas\\histogramas_cmct.txt","w")
    arqHistogramasCMCT = open("C:\\Users\\thale\\PycharmProjects\\VaiMeuFilho\\bd_projeto\\" + nomeBD + "_histogramas\\histogramas_mct8.txt", "w")

    for sujeitoNum in range(1, qtdS+1):
        qtdF = lstQtdFaces[sujeitoNum-1]
        for faceNum in range(1, qtdF+1):
            path = ('C:\\Users\\thale\\PycharmProjects\\VaiMeuFilho\\bd_projeto\\' + nomeBD + '\\s' + str(sujeitoNum) + '\\' + str(faceNum) + '.png')
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            histLBP = lbp.lbp_function(img)
            histMCT8 = mct8.mct8_function(img)
            histCMCT = cmct.cmct_function(img)

            arqHistogramasLBP.write(str(sujeitoNum) + ';' + str(faceNum) + ';' + str(histLBP) + '\n')
            arqHistogramasMCT8.write(str(sujeitoNum) + ';' + str(faceNum) + ';' + str(histMCT8) + '\n')
            arqHistogramasCMCT.write(str(sujeitoNum) + ';' + str(faceNum) + ';' + str(histCMCT) + '\n')

            print('Face ' + str(faceNum) + ' Processada')
        # end for faceNum
        print('----------------------- Sujeito ' + str(sujeitoNum) + ' Extraido \n')
    # end for sujeitoNum
    print('SALVANDO HISTOGRAMAS')

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    arqHistogramasLBP.close()
    arqHistogramasMCT8.close()
    arqHistogramasCMCT.close()

    print('PRONTO!')



if __name__ == '__main__':
	main()