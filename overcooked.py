from concurrent.futures import process, thread
from enum import Enum
from operator import attrgetter
import threading
from unicodedata import category
import numpy as np
import cv2 as cv
from mss import mss
from PIL import Image, ImageGrab
import time
from matplotlib import pyplot as plt
from main import Player
from threading import Thread
from multiprocessing import Pool, cpu_count

screen = {'left': 3960, 'top': 1555, 'width': 1920, 'height': 1080}
player = Player()


def lolol():
    print('this')


class ComputerVision:
    boolFlag = 0
    dishesToPrepare = [[[(9999, 9999)], 'none'], [[(9999, 9999)], 'none'],
                       [[(0, 0)], 'none'], [[(0, 0)], 'none']]
    dishesToPrepare2 = {}
    platePositions = []
    count = 0
    dishText = 'none'
    plates = {}
    products = {}

    def getDishPositions(self, screenshot, provisions):
        self.count += 1
        pool = Pool(processes=(cpu_count() - 1))

        for i, provision in enumerate(provisions):
            pool.apply_async(lolol)

        pool.close()
        pool.join()
        # if threading.active_count() == 1 and self.count % 5 == 0:

        #     if provision.category == "food":
        #         self.createDish(i, points, provision,
        #                         dishesToPrepareRanked, choppingBlock, self.products, self.plates)

    def getProducts(self, points):
        locationx2 = round((points[0][0] - 360) / 90) - 1
        locationy2 = round((points[0][1] - 171) / 90) - 1

        return [locationy2, locationx2]

    def getPlatePositions(self, points):
        locationx = round((points[0][0] - 360) / 90) - 1
        locationy = round((points[0][1] - 171) / 90) - 1

        return [locationy, locationx]

    def getChoppingBlockPosition(self, points):
        locationx = round((points[0][0] - 360) / 90) - 1
        locationy = round((points[0][1] - 171) / 90) - 1

        return [locationy, locationx]

    def createDish(self, i, points, dish, dishesToPrepareRanked, choppingBlock, products, plates):
        self.dishesToPrepare[i] = [points, dish.name]

        tmp = min(self.dishesToPrepare2,
                  key=self.dishesToPrepare2.get)

        if tmp == "fish":
            print(tmp)
            thread2 = Thread(target=player.prepareFish,
                             args=([plates['plate'], products['fish'], choppingBlock]))
            thread2.start()

        if tmp == "shrimp":
            print(tmp)
            thread2 = Thread(target=player.prepareShrimp,
                             args=([plates['plate'], products['shrimp'], choppingBlock]))
            thread2.start()


fishImage = cv.imread('fishDish.jpg', cv.IMREAD_UNCHANGED)
shrimpImage = cv.imread('shrimpDish.jpg', cv.IMREAD_UNCHANGED)
plateImage = cv.imread('plate.png', cv.IMREAD_UNCHANGED)

leftImage = cv.imread('leftView.jpg', cv.IMREAD_UNCHANGED)
rightImage = cv.imread('rightView.jpg', cv.IMREAD_UNCHANGED)
frontImage = cv.imread('frontView.jpg', cv.IMREAD_UNCHANGED)
backImage = cv.imread('backView.jpg', cv.IMREAD_UNCHANGED)

plate2Image = cv.imread('plate2.png', cv.IMREAD_UNCHANGED)

fishProductImage = cv.imread('fishProduct.png', cv.IMREAD_UNCHANGED)
shrimpProductImage = cv.imread('shrimpProduct.png', cv.IMREAD_UNCHANGED)

choppingBlock = cv.imread('choppingBlock.jpg', cv.IMREAD_UNCHANGED)

computerVision = ComputerVision()


class Provision:
    def __init__(self, image, name, category, threshold):
        self.image = image
        self.name = name
        self.category = category
        self.threshold = threshold


def main():
    with mss() as sct:
        frameCounter = 0

        provisions = [Provision(plateImage, 'cleanPlate', 'plate', 0.65),
                      #   Provision(plate2Image, 'cleanPlate', 'plate', 0.80),
                      #   Provision(choppingBlock, 'choppingBlock',
                      #             'preparation', 0.80),
                      #   Provision(fishImage, 'fish', 'food', 0.80),
                      #   Provision(shrimpImage, 'shrimp', 'food', 0.80),
                      #   Provision(fishProductImage, 'fish', 'product', 0.80),
                      #   Provision(shrimpProductImage, 'shrimp', 'product', 0.80),
                      #   Provision(leftImage, 'shrimp', 'product', 0.60),
                      #   Provision(rightImage, 'shrimp', 'product', 0.60),
                      #   Provision(frontImage, 'shrimp', 'product', 0.60),
                      #   Provision(backImage, 'shrimp', 'product', 0.60)


                      ]
        while True:
            frameCounter += 1
            screenshot = sct.grab(screen)
            screenshot = Image.frombytes(
                mode="RGB",
                size=(screenshot.width, screenshot.height),
                data=screenshot.rgb,
            )
            screenshot = np.flip(screenshot, axis=-1)
            screenshot = np.array(screenshot)
            computerVision.getDishPositions(screenshot, provisions)
            cv.imshow('Totally Awesome AI', screenshot)

            if cv.waitKey(1) == ord('q'):
                cv.destroyAllWindows()
                break


# thread1 = Thread(target=main)

# thread1.start()
main()
cv.destroyAllWindows()
print('destroyed')
