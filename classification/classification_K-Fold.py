import random
import numpy as np
import cv2
import ast

def leitura_arquivos(arq):
    # Realiza dump de dados dos histogramas para listas in memory
    dicHist = {}
    strLinhaArq = arq.readline()
    while (strLinhaArq != ''):                                         # remover '#'
        lstId = strLinhaArq.split(';')
        chaveSF = str(lstId[0]) + ':' + str(lstId[1])
        histograma = ast.literal_eval(lstId[2])
        dicHist[chaveSF] = histograma
        strLinhaArq = arq.readline()
    # end while
    return dicHist


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
    lstAlgoritmos = ["lbp", "mct8", "cmct"]
    lstFacesTeste = []
    for s in range(1, 41):
        lstFacesTeste.append(np.random.randint(1, 11))  # indice da chave usada para teste altera a cada sujeito lido
    #end for s

    for algoritmo in lstAlgoritmos:

        arqHistogramas = open("C:\\Users\\thale\\PycharmProjects\\VaiMeuFilho\\bd_projeto\\histograms\\histogramas_"+algoritmo+".txt", "r")
        # Estrutura: { sf: [h,i,s,t,o,g,r,a,m,a],  sf: [h,i,s,t,o,g,r,a,m,a],  sf: [h,i,s,t,o,g,r,a,m,a] }
        dicHistogramas = leitura_arquivos(arqHistogramas)
        arqHistogramas.close()

        # Arquivo -> In Memory
        lstHistTestes = []
        lstHistTreinamento = []
        for s in range(1, 41):
            for f in range(1, 11):
                keySF = str(s) + ':' + str(f)
                hist = dicHistogramas.get(keySF)
                if (f == lstFacesTeste[s-1]):
                    lstHistTestes.append([s, f, hist])
                else:
                    lstHistTreinamento.append([s, f, hist])
            # end for f
        # end for s

        # TREINAMENTO
        lstHist, lstSujeitos = separa_itens_treinamento(lstHistTreinamento)
        knn = cv2.ml.KNearest_create()
        knn.train(lstHist, cv2.ml.ROW_SAMPLE, lstSujeitos)

        # CLASSIFICAÇÃO
        arqResults = open("C:\\Users\\thale\\PycharmProjects\\VaiMeuFilho\\bd_projeto\\results\\1º_k-fold_knn_"+algoritmo+".txt", "w")
        print("---------------- TESTE "+algoritmo+" ------------------")
        for teste in lstHistTestes:
            testeSet = np.array([teste[2]], dtype=np.float32)
            resposta = teste[0]

            ret, results, neighbours, dist = knn.findNearest(testeSet, 3)
            arqResults.write("knn3: "+str(resposta)+"; "+str(int(ret))+"; "+str(results)+"; "+str(neighbours)+"; "+str(dist)+'\n')
            ret, results, neighbours, dist = knn.findNearest(testeSet, 5)
            arqResults.write("knn5: "+str(resposta)+"; "+str(int(ret))+"; "+str(results)+"; "+str(neighbours)+"; "+str(dist)+'\n')
            ret, results, neighbours, dist = knn.findNearest(testeSet, 7)
            arqResults.write("knn7: "+str(resposta)+"; "+str(int(ret))+"; "+str(results)+"; "+str(neighbours)+"; "+str(dist)+'\n')
            print("Sujeito "+ str(resposta) + " testado com "+algoritmo)
        # end for teste

        print("FIM TESTE "+algoritmo)

        arqResults.close()

    #end for algoritmo


if __name__ == '__main__':
    main()