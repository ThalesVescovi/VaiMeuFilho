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
    arqHistogramas = open("C:\\Users\\thale\\PycharmProjects\\VaiMeuFilho\\bd_projeto\\histograms\\histogramas_lbp.txt", "r")
    # Estrutura: { sf: [h,i,s,t,o,g,r,a,m,a],  sf: [h,i,s,t,o,g,r,a,m,a],  sf: [h,i,s,t,o,g,r,a,m,a] }
    dicHistogramas = leitura_arquivos(arqHistogramas)
    arqHistogramas.close()

# TODO: Verificar forma de utilizar o mesmo conjunto lstHistTestes para LBP, MCT8 e CMCT (Teste eh rapido, da pra fazer junto)

    lstHistTestes = []
    lstHistTreinamento = []
    for s in range(1, 41):
        faceTeste = np.random.randint(1, 11)  # indice da chave usada para teste altera a cada sujeito lido
        for f in range(1, 11):
            keySF = str(s) + ':' + str(f)
            hist = dicHistogramas.get(keySF)
            if (f == faceTeste):
                lstHistTestes.append([s, f, hist])
            else:
                lstHistTreinamento.append([s, f, hist])
        # end for faceNum
    # end for sujeitoNum



    # TREINAMENTO
    lstHist, lstSujeitos = separa_itens_treinamento(lstHistTreinamento)
    knn = cv2.ml.KNearest_create()
    knn.train(lstHist, cv2.ml.ROW_SAMPLE, lstSujeitos)

    arqResultsLBP = open("C:\\Users\\thale\\PycharmProjects\\VaiMeuFilho\\bd_projeto\\results\\k-fold_LBP_knn357.txt", "w")


    #CLASSIFICAÇÃO
    for teste in lstHistTestes:
        testeSet = np.array([teste[2]], dtype=np.float32)  # Em Teste: Sujeito 3; face 7;
        resposta = teste[0]

        ret, results, neighbours, dist = knn.findNearest(testeSet, 3)
        arqResultsLBP.write("knn3: "+str(resposta)+"; "+str(int(ret))+"; "+str(results)+"; "+str(neighbours)+"; "+str(dist)+'\n')
        ret, results, neighbours, dist = knn.findNearest(testeSet, 5)
        arqResultsLBP.write("knn5: " + str(resposta) + "; " + str(int(ret)) + "; " + str(results) + "; " + str(neighbours) + "; " + str(dist)+'\n')
        ret, results, neighbours, dist = knn.findNearest(testeSet, 7)
        arqResultsLBP.write("knn7: " + str(resposta) + "; " + str(int(ret)) + "; " + str(results) + "; " + str(neighbours) + "; " + str(dist)+'\n')
        print("Sujeito "+ str(resposta) + " testado com LBP.")
    # end for teste

    print("FIM TESTE LBP")


    arqResultsLBP.close()

if __name__ == '__main__':
    main()