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


def cmct_function(img):
    imgLimiarizada = img    #Inicia imgLimiarizada como a imagem comum

    media = 0
    lstHistogramaA = [0] * 256                  # De 0 a 255
    lstHistogramaB = [0] * 256                  # De 256 a 511
    lstHistogramaCMCT = [0] * 512               # União do lstHistogramaA com lstHistogramaB
    pesos = [1, 2, 4, 8, 16, 32, 64, 128]

    # Primeira extracao usando MCT8, que nos gera o lstHistogramaA e a imgLimiarizada
    for x in range(0, len(img)):
        for y in range(0, len(img[0])):
            center = img[x, y]
            top_left = get_vizinho(img, x - 1, y - 1)
            top_up = get_vizinho(img, x, y - 1)
            top_right = get_vizinho(img, x + 1, y - 1)
            left = get_vizinho(img, x - 1, y)
            right = get_vizinho(img, x + 1, y)
            bottom_left = get_vizinho(img, x - 1, y + 1)
            bottom_down = get_vizinho(img, x, y + 1)
            bottom_right = get_vizinho(img, x + 1, y + 1)
            # Verificacao horizontal com pixel central incluso na média (MCT8)
            media = (int(top_left) + int(top_up) + int(top_right) + int(right) + int(center) + int(left) + int(bottom_left) + int(bottom_down) + int(bottom_right)) // 9
            lstVizinhosBin = compara_vizinhos(media, [top_left, top_up, top_right, right, left, bottom_left, bottom_down, bottom_right])
            # Passando lista para decimal de 9 bits
            vlrDecimal = 0
            for a in range(0, len(lstVizinhosBin)):
                vlrDecimal += pesos[a] * lstVizinhosBin[a]
            # end for a
            lstHistogramaA[vlrDecimal] += 1
            imgLimiarizada.itemset((x, y), vlrDecimal)  # Transforma pixel central no vlrDecimal
        # end for y
    # end for x

    # PLOTANDO IMAGENS
    #cv2.imshow('Imagem', img)
    #cv2.imshow('Imagem Limiarizada MCT8', imgLimiarizada)

    # Extracao MCT8 em cima da imgLimiarizada, que nos gera o lstHistogramaB
    for x in range(0, len(imgLimiarizada)):
        for y in range(0, len(imgLimiarizada[0])):
            center = imgLimiarizada[x, y]
            top_left = get_vizinho(imgLimiarizada, x - 1, y - 1)
            top_up = get_vizinho(imgLimiarizada, x, y - 1)
            top_right = get_vizinho(imgLimiarizada, x + 1, y - 1)
            left = get_vizinho(imgLimiarizada, x - 1, y)
            right = get_vizinho(imgLimiarizada, x + 1, y)
            bottom_left = get_vizinho(imgLimiarizada, x - 1, y + 1)
            bottom_down = get_vizinho(imgLimiarizada, x, y + 1)
            bottom_right = get_vizinho(imgLimiarizada, x + 1, y + 1)
            # Verificacao horizontal com pixel central incluso na média (MCT8)
            media = (int(top_left) + int(top_up) + int(top_right) + int(right) + int(center) + int(left) + int(bottom_left) + int(bottom_down) + int(bottom_right)) // 9
            lstVizinhosBin = compara_vizinhos(media, [top_left, top_up, top_right, right, left, bottom_left, bottom_down, bottom_right])
            # Passando lista para decimal de 9 bits
            vlrDecimal = 0
            for a in range(0, len(lstVizinhosBin)):
                vlrDecimal += pesos[a] * lstVizinhosBin[a]
            # end for a
            lstHistogramaB[vlrDecimal] += 1
            imgLimiarizada.itemset((x, y), vlrDecimal)
        # end for y
    # end for x

    #cv2.imshow('Imagem Limiarizada CMCT', imgLimiarizada)

    # Concatena lstHistogramaA com lstHistogramaB
    lstHistogramaCMCT = lstHistogramaA + lstHistogramaB

    return lstHistogramaCMCT

