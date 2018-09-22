import numpy as np
import cv2
import algorithms.lbp as lbp
import sys
import ast
import algorithms.lbp as lbp


def leitura_arquivos(arq):
    # Realiza dump de dados dos histogramas para listas in memory
    lstHist = []
    strLinhaArq = arq.readline()
    while (strLinhaArq != ''):
        lstId = strLinhaArq.split(';')
        histograma = ast.literal_eval(lstId[2])
        lstHist.append([lstId[0], lstId[1], histograma])
        strLinhaArq = arq.readline()
    # end while
    return lstHist




def main():
    arqHistSujeitos = open("C:\\Users\\thale\\PycharmProjects\\VaiMeuFilho\\bd_projeto\\histograms\\base_histogramas_lbp.txt", "r")
    arqHistTestes = open("C:\\Users\\thale\\PycharmProjects\\VaiMeuFilho\\bd_projeto\\histograms\\teste_histogramas_lbp.txt", "r")

    # Estrutura: [ [s, f, [h,i,s,t,o,g,r,a,m,a]], [s, f, [h,i,s,t,o,g,r,a,m,a]], [s, f, [h,i,s,t,o,g,r,a,m,a]] ]
    lstHistTestes = leitura_arquivos(arqHistSujeitos)
    lstHistSujeitos = leitura_arquivos(arqHistTestes)

    arqHistSujeitos.close()
    arqHistTestes.close()







if __name__ == '__main__':
	main()