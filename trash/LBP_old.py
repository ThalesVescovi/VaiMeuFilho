import numpy as np
from matplotlib import pyplot as plt
import cv2
import psycopg2
import sys
def compara_vizinhos(center, pixels):
    lstVizinhos = []
    for pixelVizinho in pixels:
        if pixelVizinho >= center:
            lstVizinhos.append(1)
        else:
            lstVizinhos.append(0)
    return lstVizinhos


def get_vizinho(img, idx, idy, default=0):
    try:
        return img[idx, idy]
    except IndexError:

        return default





def main():
    #imgcolorida = cv2.imread('C:\\Users\\thale\\PycharmProjects\\VaiMeuFilho\\images\\arnie_20_20_200_200.jpg', cv2.IMREAD_COLOR)
    img = cv2.imread('C:\\Users\\thale\\PycharmProjects\\VaiMeuFilho\\trash\\images\\arnie_20_20_200_200.jpg', cv2.IMREAD_GRAYSCALE)
    imgLimiarizada = cv2.imread('C:\\Users\\thale\\PycharmProjects\\VaiMeuFilho\\trash\\images\\arnie_20_20_200_200.jpg', cv2.IMREAD_GRAYSCALE)

    lstHistograma = [0] * 256
    pesos = [1, 2, 4, 8, 16, 32, 64, 128]

    for x in range(0, len(img)):
        for y in range(0, len(img[0])):
            center = img[x, y]
            top_left = get_vizinho(img, x-1, y-1)
            top_up = get_vizinho(img, x, y-1)
            top_right = get_vizinho(img, x+1, y-1)
            right = get_vizinho(img, x+1, y)
            bottom_right = get_vizinho(img, x+1, y+1)
            bottom_down = get_vizinho(img, x, y+1)
            bottom_left = get_vizinho(img, x-1, y+1)
            left = get_vizinho(img, x-1, y)
                                                                             # Verificaçao circular
            lstVizinhosBin = compara_vizinhos(center, [top_left, top_up, top_right, right, bottom_right, bottom_down, bottom_left, left])
            #PASSANDO LISTA PARA DECIMAL
            vlrDecimal = 0
            for a in range(0, len(lstVizinhosBin)):
                vlrDecimal += pesos[a] * lstVizinhosBin[a]

            lstHistograma[vlrDecimal] += 1
            imgLimiarizada.itemset((x, y), vlrDecimal) #transforma pixel central no valor decimal
            #print(str(lstValues)+' -> '+str(decimal))

        #end_for
    #end_for

    print(len(lstHistograma))
    print(lstHistograma)

    #print(str(lstHistograma))




    #IMPRIMINDO HISTOGRAMA LBP
    #histLBP = np.histogram(imgLimiarizada.flatten(), 256, [0, 256])  # Histograma da imagem limiarizada
    #print(histLBP[0])  # Vetor Histograma
    # print(histLBP[1]) #Posicoes do vetor


    #PLOTANDO IMAGENS
    #cv2.imshow('Imagem Colorida', imgcolorida)
    #cv2.imshow('Imagem', img)
    #cv2.imshow('Imagem Limiarizada', imgLimiarizada)


    #PLOTAND GRÁFICO HISTOGRAMA
    #hist, bins = np.histogram(img.flatten(), 256, [0, 256])  # Histograma da imagem comum
    #cdf = hist.cumsum()
    #cdf_normalized = cdf * hist.max() / cdf.max()
    #plt.plot(cdf_normalized, color='b')

    #plt.hist(imgLimiarizada.flatten(), 256, [0, 256], color='r')
    #plt.xlim([0, 256])
    #plt.legend(('cdf', 'histogram'), loc= 'upper left')
    #plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()






if __name__ == '__main__':
	main()