import random
import numpy as np
import cv2
import ast

def leitura_arquivos(arq):
    # Realiza dump de dados dos histogramas para listas in memory
    lstHist = []
    strLinhaArq = arq.readline()
    while (strLinhaArq != ''):                                         # remover '#'
        lstId = strLinhaArq.split(';')
        histograma = ast.literal_eval(lstId[2])
        lstHist.append([lstId[0], lstId[1], histograma])
        strLinhaArq = arq.readline()
    # end while
    return lstHist


def separa_itens_treinamento(lstArqHistSuj):
    lstHist = []
    lstSujeitoAux = []
    lstSujeitos = []
    for row in range(0, len(lstArqHistSuj)):
        lstHist.append(lstArqHistSuj[row][2])
        lstSujeitoAux.append(int(lstArqHistSuj[row][0]))
        lstSujeitos.append(lstSujeitoAux)
        lstSujeitoAux = []
    #end for
    return np.array(lstHist, dtype=np.float32), np.array(lstSujeitos, dtype=np.float32)



def main():
    arqHistSujeitos = open("C:\\Users\\thale\\PycharmProjects\\VaiMeuFilho\\bd_projeto\\histograms\\base_histogramas_lbp.txt", "r")
    arqHistTestes = open("C:\\Users\\thale\\PycharmProjects\\VaiMeuFilho\\bd_projeto\\histograms\\teste_histogramas_lbp.txt", "r")
    # Estrutura: [ [s, f, [h,i,s,t,o,g,r,a,m,a]], [s, f, [h,i,s,t,o,g,r,a,m,a]], [s, f, [h,i,s,t,o,g,r,a,m,a]] ]
    lstArqHistSujeitos = leitura_arquivos(arqHistSujeitos)
    lstArqHistTestes = leitura_arquivos(arqHistTestes)

    lstHist, lstSujeitos = separa_itens_treinamento(lstArqHistSujeitos)

    # TREINAMENTO
    knn = cv2.ml.KNearest_create()
    knn.train(lstHist, cv2.ml.ROW_SAMPLE, lstSujeitos)
    print(lstHist)
    print(lstSujeitos)

    #CLASSIFICAÇÃO          lstArqHistTestes[14-1][2] = histograma
    testeSet = np.array([lstArqHistTestes[3-1][2]], dtype=np.float32)  # Em Teste: Sujeito 3; face 7;
    ret, results, neighbours, dist = knn.findNearest(testeSet, 5)
    print("ret: ", ret, "\n")
    print("result: ", results, "\n")
    print("neighbours: ", neighbours, "\n")
    print("distance: ", dist)


    arqHistSujeitos.close()
    arqHistTestes.close()


if __name__ == '__main__':
    main()