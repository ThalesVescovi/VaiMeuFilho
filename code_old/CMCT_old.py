import numpy as np
from matplotlib import pyplot as plt
import cv2

def compara_vizinhos(media, pixels):
    lstVizinhos = []
    for pixelVizinho in pixels:
        if pixelVizinho >= media:
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
    img = cv2.imread('C:\\Users\\thale\\PycharmProjects\\VaiMeuFilho\\images\\arnie_20_20_200_200.jpg', cv2.IMREAD_GRAYSCALE)
    imgLimiarizada = cv2.imread('C:\\Users\\thale\\PycharmProjects\\VaiMeuFilho\\images\\arnie_20_20_200_200.jpg', cv2.IMREAD_GRAYSCALE)

    media = 0
    lstHistograma = [0]*512
    pesos = [256, 128, 64, 32, 16, 8, 4, 2, 1]

    for x in range(0, len(img)):
        for y in range(0, len(img[0])):
            center = img[x, y]
            top_left = get_vizinho(img, x-1, y-1)
            top_up = get_vizinho(img, x, y-1)
            top_right = get_vizinho(img, x+1, y-1)
            left = get_vizinho(img, x-1, y)
            right = get_vizinho(img, x+1, y)
            bottom_left = get_vizinho(img, x-1, y+1)
            bottom_down = get_vizinho(img, x, y+1)
            bottom_right = get_vizinho(img, x+1, y+1)

            media = (int(top_left) + int(top_up) + int(top_right) + int(right) + int(center) + int(left) + int(bottom_left) + int(bottom_down) + int(bottom_right) ) // 9
            lstVizinhosBin = comparacompara_vizinhosVizinhos(media, [top_left, top_up, top_right, right, center, left, bottom_left, bottom_down, bottom_right])

            #print(str(top_left)+' '+str(top_up)+' '+str(top_right)+' '+str(right)+' '+str(center)+' '+str(left)+' '+str(bottom_left)+' '+str(bottom_down)+' '+str(bottom_right))
            #print(media)
            #print(lstVizinhosBin)

            # PASSANDO LISTA PARA DECIMAL
            vlrDecimal = 0
            for a in range(0, len(lstVizinhosBin)):
                vlrDecimal += pesos[a] * lstVizinhosBin[a]

            lstHistograma[vlrDecimal] += 1
            imgLimiarizada.itemset((x, y), vlrDecimal) # Transforma pixel central no valor decimal
            #print(str(lstValues)+' -> '+str(decimal))

        #end_for
    #end_for

    print(len(lstHistograma))
    print(lstHistograma)

    # IMPRIMINDO HISTOGRAMA LBP
    histCMCT = np.histogram(imgLimiarizada.flatten(), 512, [0, 512])  # Histograma da imagem limiarizada
    print(histCMCT[0])  # Vetor Histograma

    #print(histLBP[1])  # Posicoes do vetor

    # PLOTANDO IMAGENS
    #cv2.imshow('Imagem Colorida', imgcolorida)
    #cv2.imshow('Imagem', img)
    #cv2.imshow('Imagem Limiarizada', imgLimiarizada)


    # PLOTAND GRÁFICO HISTOGRAMA
    hist, bins = np.histogram(img.flatten(), 512, [0, 512])  # Histograma da imagem comum
    #cdf = hist.cumsum()
    #cdf_normalized = cdf * hist.max() / cdf.max()
    #plt.plot(cdf_normalized, color='b')

    plt.hist(imgLimiarizada.flatten(), 512, [0, 512], color='r')
    #plt.hist(lstHistograma, 512, [0, 512], color='r')
    plt.xlim([0, 512])
    plt.legend(('cdf', 'histogram'), loc= 'upper left')
    plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == '__main__':
	main()