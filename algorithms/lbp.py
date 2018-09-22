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


def lbp_function(img):
    lstHistograma = [0] * 256
    pesos = [1, 2, 4, 8, 16, 32, 64, 128]
    for x in range(0, len(img)):
        for y in range(0, len(img[0])):
            center = img[x, y]
            top_left = get_vizinho(img, x - 1, y - 1)
            top_up = get_vizinho(img, x, y - 1)
            top_right = get_vizinho(img, x + 1, y - 1)
            right = get_vizinho(img, x + 1, y)
            bottom_right = get_vizinho(img, x + 1, y + 1)
            bottom_down = get_vizinho(img, x, y + 1)
            bottom_left = get_vizinho(img, x - 1, y + 1)
            left = get_vizinho(img, x - 1, y)
            # Verificacao circular
            lstVizinhosBin = compara_vizinhos(center, [top_left, top_up, top_right, right, bottom_right, bottom_down, bottom_left, left])
            # Passando lista para decimal
            vlrDecimal = 0
            for a in range(0, len(lstVizinhosBin)):
                vlrDecimal += pesos[a] * lstVizinhosBin[a]
            #end for a
            lstHistograma[vlrDecimal] += 1
        #end for y
    #end for x
    return lstHistograma