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
    media = 0
    lstHistograma = [0] * 512
    pesos = [256, 128, 64, 32, 16, 8, 4, 2, 1]
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
            # Verificacao horizontal com pixel central incluso na média
            media = (int(top_left) + int(top_up) + int(top_right) + int(right) + int(center) + int(left) + int(bottom_left) + int(bottom_down) + int(bottom_right)) // 9
            lstVizinhosBin = compara_vizinhos(media, [top_left, top_up, top_right, right, center, left, bottom_left, bottom_down, bottom_right])
            # Passando lista para decimal de 9 bits
            vlrDecimal = 0
            for a in range(0, len(lstVizinhosBin)):
                vlrDecimal += pesos[a] * lstVizinhosBin[a]
            # end for a
            lstHistograma[vlrDecimal] += 1
        # end for y
    # end for x
    return lstHistograma
