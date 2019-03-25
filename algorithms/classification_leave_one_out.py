import random
import numpy as np
import cv2
import ast


def leitura_arquivos(arq):
    # Realiza dump de dados dos histogramas para listas in memory
    dicHist = {}
    strLinhaArq = arq.readline()
    while (strLinhaArq != ''):  # remover '#'
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
    # end for
    return np.array(lstHist, dtype=np.float32), np.array(lstSujeitos, dtype=np.float32)


def main():
    # PARAMETROS PARA LEITURA DOS ARQUIVOS
    #nomeBD = "att_faces"; qtdS = 40; lstQtdFaces = [10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,]
    #nomeBD = "umist"; qtdS = 20; lstQtdFaces = [38,35,26,24,26,23,19,22,20,32,34,34,26,30,19,26,26,33,48,34]
    nomeBD = "jaffe"; qtdS = 10; lstQtdFaces = [23,22,22,20,21,21,20,21,21,22]


    lstAlgoritmos = ["lbp", "mct8", "cmct"]
    numTotalFaces = sum(lstQtdFaces)
    for algoritmo in lstAlgoritmos:
        acertosknn3 = 0
        acertosknn5 = 0
        acertosknn7 = 0

        arqHistogramas = open("C:\\Users\\thale\\PycharmProjects\\VaiMeuFilho\\bd_projeto\\" + nomeBD + "_histogramas\\histogramas_" + algoritmo + ".txt", "r")
        # Estrutura: { sf: [h,i,s,t,o,g,r,a,m,a],  sf: [h,i,s,t,o,g,r,a,m,a],  sf: [h,i,s,t,o,g,r,a,m,a] }
        dicHistogramas = leitura_arquivos(arqHistogramas)
        arqHistogramas.close()

        arqResults = open("C:\\Users\\thale\\PycharmProjects\\VaiMeuFilho\\bd_projeto\\" + nomeBD + "_results\\leave-one-out_knn_" + algoritmo + ".txt", "w")
        print("---------------- TESTE " + algoritmo + " ------------------")

        for faceTeste in range(1, numTotalFaces+1):
            faceCorrente = 0

            # Arquivo -> In Memory
            histTestes = []
            lstHistTreinamento = []
            for s in range(1, qtdS+1):
                qtdF = lstQtdFaces[s - 1]
                for f in range(1, qtdF+1):
                    faceCorrente += 1
                    keySF = str(s) + ':' + str(f)
                    hist = dicHistogramas.get(keySF)
                    if (faceCorrente == faceTeste):
                        histTestes = [s, f, hist]
                    else:
                        lstHistTreinamento.append([s, f, hist])
                # end for f
            # end for s

            # TREINAMENTO
            lstHist, lstSujeitos = separa_itens_treinamento(lstHistTreinamento)
            knn = cv2.ml.KNearest_create()
            knn.train(lstHist, cv2.ml.ROW_SAMPLE, lstSujeitos)

            # CLASSIFICAÇÃO
            testeSet = np.array([histTestes[2]], dtype=np.float32)
            resposta = histTestes[0]

            ret, results, neighbours, dist = knn.findNearest(testeSet, 3)
            if (str(int(ret)) == str(resposta)) : acertosknn3 += 1
            arqResults.write("(Face "+str(faceTeste)+") knn3: "+str(resposta)+ "; "+str(int(ret))+"; "+str(results)+"; "+str(neighbours)+"; "+str(dist)+'\n')

            ret, results, neighbours, dist = knn.findNearest(testeSet, 5)
            if (str(int(ret)) == str(resposta)): acertosknn5 += 1
            arqResults.write("(Face "+str(faceTeste)+") knn5: "+str(resposta)+ "; "+str(int(ret))+"; "+str(results)+"; "+str(neighbours)+"; "+str(dist)+'\n')

            ret, results, neighbours, dist = knn.findNearest(testeSet, 7)
            if (str(int(ret)) == str(resposta)): acertosknn7 += 1
            arqResults.write("(Face "+str(faceTeste)+") knn7: "+str(resposta)+ "; "+str(int(ret))+"; "+str(results)+"; "+str(neighbours)+"; "+str(dist)+'\n')
            #print("Face " + str(faceTeste) + " testado com " + algoritmo)
        #end for faceTeste

        print("(" + str(algoritmo) + ") " + "knn3: " + str(acertosknn3) + " acertos de "+str(numTotalFaces)+".  Acurácia: " + str((acertosknn3 / numTotalFaces) * 100) + "%")
        print("(" + str(algoritmo) + ") " + "knn5: " + str(acertosknn5) + " acertos de "+str(numTotalFaces)+".  Acurácia: " + str((acertosknn5 / numTotalFaces) * 100) + "%")
        print("(" + str(algoritmo) + ") " + "knn7: " + str(acertosknn7) + " acertos de "+str(numTotalFaces)+".  Acurácia: " + str((acertosknn7 / numTotalFaces) * 100) + "%")
        print("\n")

        arqResults.close()
    #end for algoritmo


if __name__ == '__main__':
    main()