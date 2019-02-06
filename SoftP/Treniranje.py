import cv2
import numpy as np
import matplotlib.pyplot as plt
import collections
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD
from PIL import Image
from scipy import signal
import glob
from keras.utils import np_utils
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
from keras.layers import Flatten, Conv2D, MaxPool2D
import matplotlib.pylab as pylab
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from PIL import Image
import random


def invert(image):
    return 255 - image


def dilate(image):
    kernel = np.ones((3, 3))
    return cv2.dilate(image, kernel, iterations=2)


def shift_right(img):
    rows, cols = img.shape
    M = np.float32([[1, 0, 3], [0, 1, 0]])
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst


def shift_left(img):
    rows, cols = img.shape
    M = np.float32([[1, 0, -3], [0, 1, 0]])
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst


def rotate_left(img):
    rows, cols = img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 10, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst


def rotate_right(img):
    rows, cols = img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -10, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst


def erode(image):
    kernel = np.ones((3, 3))
    return cv2.erode(image, kernel, iterations=1)


def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def image_otsu(blur):
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th3


def image_bin(image_gs):
    ret, image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    return image_bin


def display_image(image):
    plt.imshow(image)
    plt.show()


def izracunaj_k(x1, y1, x2, y2):
    return (y2 - y1) / (x2 - x1)


def izracunaj_n(x, y, k):
    return y - (x * k)


def izracunaj_y(x, k, n):
    return (x * k) + n


def select_roi(image_orig, image_bin):
    contours,img= cv2.findContours(image_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)
    sorted_regions = []  # lista sortiranih regiona po x osi (sa leva na desno)
    regions_array = []
    regions_coord = []
    regions_coord1 = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # koordinate i velicina granicnog pravougaonika
        area = cv2.contourArea(contour)
        if (area > 12 and h < 40 and h > 15 and w > 0.3) or (w > 2 and h > 14 and h < 40 and area > 13) or (
                w > 13 and h > 9 and h < 60 and area > 13):
            # kopirati [y:y+h+1, x:x+w+1] sa binarne slike i smestiti u novu sliku
            # označiti region pravougaonikom na originalnoj slici (image_orig) sa rectangle funkcijom
            image_g = image_gray(image_orig)
            image_blur = cv2.GaussianBlur(image_g, (3, 3), 0)
            i_otsu = image_otsu(image_blur)
            region = image_g[y - 6:y + h + 5, x - round((28 - w) / 2) - 1:x + w + round((28 - w) / 2 + 1)]
            # region = (dilate(erode(region)))

            # region = cv2.resize(region,(280,280), interpolation = cv2.INTER_NEAREST)
            # region = erode(dilate(region))
            region = resize_region(region)
            # CISCENJE
            k = 0
            for i in range(28, 19, -1):
                for j in range(16 + k, 29):
                    region[i - 1][j - 1] = 0
                k = k + 1
            for i in range(0, 15):
                for j in range(0, 18 - i):
                    region[j][i] = 0

            for i in range(0, 28):
                region[i][0] = 0
                region[i][1] = 0
                region[i][26] = 0
                region[i][27] = 0
            # CISCENJE
            regions_coord.append([(x - 7, y - 7), (x, y)])
            # region = erode(dilate(region))
            regions_array.append([region, (x, y, w, h)])
            cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)
    regions_array = sorted(regions_array, key=lambda item: item[1][0])
    regions_coord = sorted(regions_coord, key=lambda item: item[1][0])
    sorted_regions = sorted_regions = [region[0] for region in regions_array]
    regions_coord1 = regions_coord1 = [coord[0] for coord in regions_coord]

    # sortirati sve regione po x osi (sa leva na desno) i smestiti u promenljivu sorted_regions
    return image_orig, sorted_regions, regions_coord1


def resize_region(region):
    return cv2.resize(region, (28, 28), interpolation=cv2.INTER_NEAREST)


def scale_to_range(image):  # skalira elemente slike na opseg od 0 do 1
    return image / 255


def matrix_to_vector(image):
    return image.flatten()


def prepare_for_ann(regions):
    ready_for_ann = []
    for region in regions:
        scale = scale_to_range(region)
        ready_for_ann.append(matrix_to_vector(scale))
    return ready_for_ann


def convert_output(alphabet):
    nn_outputs = []
    for index in range(len(alphabet)):
        output = np.zeros(len(alphabet))
        output[index] = 1
        nn_outputs.append(output)
    return np.array(nn_outputs)


def create_ann():
    ann = Sequential()

    ann.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                   activation='relu', input_shape=(28, 28, 1)))
    ann.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                   activation='relu'))
    ann.add(MaxPool2D(pool_size=(2, 2)))
    ann.add(Dropout(0.25))

    ann.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                   activation='relu'))
    ann.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                   activation='relu'))
    ann.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    ann.add(Dropout(0.25))
    ann.add(Flatten())
    ann.add(Dense(256, activation="relu"))
    ann.add(Dropout(0.5))
    ann.add(Dense(10, activation="softmax"))
    return ann


def create_ann1():
    ann = Sequential()
    ann.add(Dense(128, input_dim=784, activation='relu'))
    ann.add(Dropout(0.2))
    ann.add(Dense(128))
    ann.add(Activation('relu'))
    #ann.add(Dropout(0.2))
    ann.add(Dense(10, activation='softmax'))
    return ann


def train_ann(ann, X_train, y_train):
    X_train = np.array(X_train, np.float32)
    y_train = np.array(y_train, np.float32)
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    sgd = SGD(lr=0.01, momentum=0.9)
    ann.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
    ann.fit(X_train, y_train, epochs=60, batch_size=256, verbose=1, shuffle=True)
    return ann


def winner(output):
    return max(enumerate(output), key=lambda x: x[1])[0]


def display_result(outputs, alphabet):
    result = []
    for output in outputs:
        result.append(alphabet[winner(output)])
    return result


(X_train, y_train), (X_test, y_test) = mnist.load_data()
alphabet = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(X_train.shape)

ann = create_ann1()


nb_classes = 10
from keras.utils import np_utils

#DATA AUGMENTATION

for i in range(0, 5000):
    X_train[1000+i] = rotate_right(X_train[1000+i])
for j in range(0, 5000):
    X_train[11000+j] = rotate_left(X_train[11000+j])
for k in range(0, 5000):
    X_train[21000+k] = shift_right(X_train[21000+k])
for l in range(0, 5000):
    X_train[31000+l] = shift_left(X_train[31000+l])


X_train = prepare_for_ann(X_train)
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

#DATA AUGMENTATION

ann = train_ann(ann, X_train, Y_train)

test_skup = X_test[0:50]
test_rez = Y_test[0:50]

test_skup = prepare_for_ann(test_skup)
result = ann.predict(np.array(test_skup, np.float32))
print(result)
print(display_result(result, alphabet))
print(display_result(test_rez[0:50], alphabet))

zbirovi = []
for iii in range(0,10):
    strTemp = str(iii)
    cap = cv2.VideoCapture('videos/video-'+strTemp+'.avi')
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    currentFrame = 0
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # plt.imshow(frame)
    # plt.show()

    lower = np.array([0, 200, 0])
    upper = np.array([100, 255, 100])

    maska = cv2.inRange(frame, lower, upper)
    # plt.imshow(maska,'gray')
    # plt.show()

    edges = cv2.Canny(maska, 75, 150, apertureSize=5)
    linesG = cv2.HoughLinesP(edges, 1, np.pi / 180, 60, maxLineGap=20, minLineLength=5)
    # print(linesG)

    okvir = edges.copy()
    okvir = testBL = cv2.GaussianBlur(okvir, (3, 3), 0)
    okvir = dilate(erode(erode(dilate(erode(dilate(okvir))))))
    ret, okvir = cv2.threshold(okvir, 70, 255, cv2.THRESH_BINARY)

    indices = np.where(okvir == [255])
    # print(indices)
    coordinates = zip(indices[0], indices[1])

    pointsG = []
    y = indices[0]
    x = indices[1]
    pointsG.append(x)
    pointsG.append(y)

    # plt.imshow(okvir, 'gray')
    # plt.show()

    lower = np.array([0, 0, 200])
    upper = np.array([100, 100, 255])

    maska = cv2.inRange(frame, lower, upper)

    edges = cv2.Canny(maska, 75, 150, apertureSize=5)
    linesB = cv2.HoughLinesP(edges, 1, np.pi / 180, 60, maxLineGap=20, minLineLength=5)
    # print(linesB)

    okvir = edges.copy()
    okvir = erode(dilate(okvir))

    indices = np.where(okvir == [255])
    # print(indices)
    coordinates = zip(indices[0], indices[1])

    pointsB = []
    y = indices[0]
    x = indices[1]
    pointsB.append(x)
    pointsB.append(y)

    # plt.imshow(okvir, 'gray')
    # plt.show()

    img_list = []

    currentFrame = 1
    zbir = 0
    cooldownB = 0
    cooldownG = 0
    pppposlednji_B = 0
    pppposlednji_G = 0
    predposlednji_B = 0
    predposlednji_G = 0
    poslednji_B = 0
    poslednji_G = 0




    while (currentFrame < length):
        #     if currentFrame == 401:
        #         break

        # time.sleep(0.5)
        ret, frame = cap.read()
        # frame = shift_right(frame)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # name = './slike/slika'+'9-'+str(currentFrame) +'.jpg'
        # print( currentFrame)
        currentFrame += 1
        if (currentFrame <= 20):
            continue
        # print('Izrada...'+name)
        # cv2.imwrite(name,frame);
        x1b = 700
        x2b = 0
        y1b = 0
        y2b = 700
        x1g = 700
        x2g = 0
        y1g = 0
        y2g = 700

        for line in linesG:
            x1, y1, x2, y2 = line[0]
            if x1 < x1g:
                x1g = x1
            if x2 > x2g:
                x2g = x2
            if y1 > y1g:
                y1g = y1
            if y2 < y2g:
                y2g = y2
            lm = line[0]
            # cv2.line(frame, (x1, y1), (x2, y2), (150, 150, 0), 2)
        cv2.line(frame, (x1g, y1g), (x2g, y2g), (150, 150, 0), 2)

        for line in linesB:
            x1, y1, x2, y2 = line[0]
            if x1 < x1b:
                x1b = x1
            if x2 > x2b:
                x2b = x2
            if y1 > y1b:
                y1b = y1
            if y2 < y2b:
                y2b = y2
            lp = line[0]
            # cv2.line(frame, (x1, y1), (x2, y2), (0, 150, 150), 2)
        cv2.line(frame, (x1b, y1b), (x2b, y2b), (0, 150, 150), 2)
        #     print(x1b)
        #     print(x2b)
        #     print(y1b)
        #     print(y2b)
        testI = frame.copy()

        # TEST SLIKA
        # testI = cv2.imread('Desktop/SoftP/slike/slika0-228.jpg')

        ##BLUR MI UNISTI JEDAN BROJ ???
        testBL = cv2.GaussianBlur(testI, (3, 3), 0)

        testC = frame.copy()
        # testC = cv2.addWeighted(testC,1.9,np.zeros(testC.shape,testC.dtype),0,10)
        # testC = dilate(erode(testC))
        # testC = shift_right(testC)

        testI = cv2.cvtColor(testI, cv2.COLOR_BGR2RGB)

        # testI = image_bin(image_gray(testI))
        lower = np.array([90, 90, 90])
        upper = np.array([255, 255, 255])

        testI = cv2.inRange(testI, lower, upper)

        # plt.imshow(testI, 'gray')
        # plt.show()
        testB = dilate(erode(testI))
        testB1 = erode(dilate(testI))
        # plt.imshow(testB, 'gray')
        # plt.show()
        try:
            selected_regions, numbers, regions_coord = select_roi(testC, testI)
            cv2.imshow("X",selected_regions)
        except:
            continue
        # display_image(selected_regions)

        l = 0
        j = 0
        i = len(numbers)

        #     for number in numbers:
        #         plt.imshow(number,'gray')
        #         plt.show()

        ## PRESEK SA LINIJOM

        # if cooldownB == 0:
        for coord in regions_coord:

            #         preseko = 0
            #         k= 0
            k = izracunaj_k(x1b, y1b, x2b, y2b)
            n = izracunaj_n(x1b, y1b, k)
            yn = k * coord[0] + n

            if cooldownB == 0 or (abs(poslednji_B - yn) > 4 and abs(predposlednji_B - yn) > 4 and abs(pppposlednji_B - yn) > 4):
                if abs(yn - coord[1]) <= 1.3 and (coord[0] >= x1b and coord[0] <= x2b) and (
                        coord[1] <= y1b and coord[1] >= y2b):
                    #                 print(yn)
                    #                 print(poslednji_B)
                    #                 print("COOLDOWNB")
                    #                 print(cooldownB)
                    cooldownB = 7
                    # for number in numbers:
                    pppposlednji_B = predposlednji_B
                    predposlednji_B = poslednji_B
                    poslednji_B = yn

                    #display_image(selected_regions)
                    # print(coord)
                    #             print("JJJ =")
                    #             print(j)
                    # print("YYY =")
                    # print(yn)
                    # print(coord[1])
                    #plt.imshow(numbers[j], 'gray')
                    #plt.show()
                    pom = np.array([numbers[j]])
                    pom1 = prepare_for_ann(pom)
                    pom2 = ann.predict(np.array(pom1, np.float32))
                    preseko = 0
                    # print(pom2)
                    print("SABERI SA:")
                    print(display_result(pom2, alphabet))
                    zbir = zbir + display_result(pom2, alphabet)[0]

                    print("UKUPNO:::")
                    print("= ")
                    print(zbir)
            j += 1

        # if cooldownG == 0:
        for coord in regions_coord:

            #         preseko = 0
            #         k= 0
            k = izracunaj_k(x1g, y1g, x2g, y2g)
            n = izracunaj_n(x1g, y1g, k)
            yn = k * coord[0] + n

            if cooldownG == 0 or (abs(poslednji_G - yn) > 4 and abs(predposlednji_G - yn) > 4 and abs(pppposlednji_G - yn) > 4):
                if abs(yn - coord[1]) <= 1.3 and (coord[0] >= x1g and coord[0] <= x2g) and (
                        coord[1] <= y1g and coord[1] >= y2g):
                    #                 print(yn)
                    #                 print(poslednji_G)
                    #                 print("COOLDOWNG")
                    #                 print(cooldownG)
                    cooldownG = 7
                    # for number in numbers:
                    pppposlednji_G = predposlednji_G
                    predposlednji_G = poslednji_G
                    poslednji_G = yn
                    #display_image(selected_regions)
                    # print(coord)
                    #                 print("JJJ =")
                    #                 print(j)
                    # print("YYY =")
                    # print(yn)
                    # print(coord[1])
                    #plt.imshow(numbers[l], 'gray')
                    #plt.show()
                    pom = np.array([numbers[l]])
                    pom1 = prepare_for_ann(pom)
                    pom2 = ann.predict(np.array(pom1, np.float32))
                    preseko = 0
                    # print(pom2)
                    print("ODUZMI SA:")
                    print(display_result(pom2, alphabet))
                    zbir = zbir - display_result(pom2, alphabet)[0]

                    print("UKUPNO:::")
                    print("= ")
                    print(zbir)
            l += 1
        #         for pointBx in pointsB[0]:
        #             if cooldown != 0:
        #                     cooldown -=1;
        #             if abs(pointBx-coord[0])<1 and abs(pointsB[1][k]- coord[1])<1:
        #                 print(pointBx)
        #                 print(coord[0])
        #                 print(pointsB[1][k])
        #                 print(coord[1])

        #                 if cooldown == 0:
        #                     preseko = 1
        #                     cooldown = 6
        #                     break
        #             k+=1
        #         if preseko == 1:
        #             pom = np.array([numbers[j]])
        #             pom1 = prepare_for_ann(pom)
        #             pom2 = ann.predict(np.array(pom1, np.float32))
        #             preseko = 0
        #             print(pom2)
        #             print(display_result(pom2,alphabet))

        #             zbir = zbir+ display_result(pom2,alphabet)[0]
        #         j+=1

        if cooldownG != 0:
            cooldownG -= 1

        if cooldownB != 0:
            cooldownB -= 1
        # print("ZBIIIIR:::")
        # print("= ")
        # print(zbir)
        # np.set_printoptions(threshold=np.nan)
        # print(pointsB)
        #     if currentFrame == 400:
        #         break
        # display_image(frame)
        # cv2.imshow("frame", frame)
        # img_list.append(frame)

        key = cv2.waitKey(25)

    cap.release()
    print("ZBIR!!!!!!")
    print(zbir)
    zbirovi.append(zbir)

print(zbirovi)