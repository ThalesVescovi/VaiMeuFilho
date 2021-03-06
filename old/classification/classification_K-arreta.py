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
    lstAccuknn3 = []
    lstAccuknn5 = []
    lstAccuknn7 = []

    for ciclo in range(1, 6):
        lstAlgoritmos = ["lbp", "mct8", "cmct"]
        lstFacesTeste = []
        for s in range(1, 41):
            lstFacesTeste.append(np.random.randint(1, 11))  # indice da chave usada para teste altera a cada sujeito lido
        #end for s

        for algoritmo in lstAlgoritmos:
            acertosknn3 = 0
            acertosknn5 = 0
            acertosknn7 = 0

            arqHistogramas = open("C:\\Users\\thale\\PycharmProjects\\VaiMeuFilho\\bd_projeto\\att_faces_histogramas\\histogramas_"+algoritmo+".txt", "r")
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
            arqResults = open("C:\\Users\\thale\\PycharmProjects\\VaiMeuFilho\\bd_projeto\\att_faces_results\\"+str(ciclo)+"º_k-arreta_knn_"+algoritmo+".txt", "w")
            print("---------------- "+str(ciclo)+"º TESTE "+algoritmo+" ------------------")
            for teste in lstHistTestes:
                testeSet = np.array([teste[2]], dtype=np.float32)
                resposta = teste[0]

                ret, results, neighbours, dist = knn.findNearest(testeSet, 3)
                if (str(int(ret)) == str(resposta)): acertosknn3 += 1
                arqResults.write("knn3: "+str(resposta)+"; "+str(int(ret))+"; "+str(results)+"; "+str(neighbours)+"; "+str(dist)+'\n')

                ret, results, neighbours, dist = knn.findNearest(testeSet, 5)
                if (str(int(ret)) == str(resposta)): acertosknn5 += 1
                arqResults.write("knn5: "+str(resposta)+"; "+str(int(ret))+"; "+str(results)+"; "+str(neighbours)+"; "+str(dist)+'\n')

                ret, results, neighbours, dist = knn.findNearest(testeSet, 7)
                if (str(int(ret)) == str(resposta)): acertosknn7 += 1
                arqResults.write("knn7: "+str(resposta)+"; "+str(int(ret))+"; "+str(results)+"; "+str(neighbours)+"; "+str(dist)+'\n')
            # end for teste

            lstAccuknn3.append((acertosknn3 / 40) * 100)
            print("(" + str(algoritmo) + ") " + "knn3: " + str(acertosknn3) + " acertos de 40.  Acurácia: " + str((acertosknn3/40)*100) + "%")
            lstAccuknn5.append((acertosknn5 / 40) * 100)
            print("(" + str(algoritmo) + ") " + "knn5: " + str(acertosknn5) + " acertos de 40.  Acurácia: " + str((acertosknn5/40)*100) + "%")
            lstAccuknn7.append((acertosknn7 / 40) * 100)
            print("(" + str(algoritmo) + ") " + "knn7: " + str(acertosknn7) + " acertos de 40.  Acurácia: " + str((acertosknn7/40)*100) + "%")

            print("\n")

            arqResults.close()
        #end for algoritmo

    #end for ciclo


if __name__ == '__main__':
    main()