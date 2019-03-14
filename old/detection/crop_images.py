import cv2
import sys
import os


def faceCropper(image_path, show_result):
    CASCADE_PATH = "C:\\Users\\thale\\PycharmProjects\\VaiMeuFilho\\detection\\haarcascade_frontalface_alt.xml"
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    img = cv2.imread(image_path)
    if (img is None):
        print("Can't open image file")
        return 0

    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img, 1.1, 3, minSize=(100, 100))
    if (faces is None):
        print('Failed to detect face')
        return 0

    if (show_result):
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    facecnt = len(faces)
    print("Detected faces: %d" % facecnt)
    i = 0
    height, width = img.shape[:2]

    for (x, y, w, h) in faces:
        r = max(w, h) / 2
        centerx = x + w / 2
        centery = y + h / 2
        nx = int(centerx - r)
        ny = int(centery - r)
        nr = int(r * 2)

        faceimg = img[ny:ny+nr, nx:nx+nr]
        lastimg = cv2.resize(faceimg, (32, 32))
        i += 1
        cv2.imwrite("image%d.jpg" % i, lastimg)

        pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
        pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(img, M, (300, 300))
#end faceCropper

        cv2.imshow('Imagem Limiarizada CMCT', imgLimiarizada)

def main():
    path = ('C:\\Users\\thale\\PycharmProjects\\VaiMeuFilho\\bd_projeto\\att_faces\\s1\\3.pgm')
    faceCropper(path, True)





if __name__ == '__main__':
    main()