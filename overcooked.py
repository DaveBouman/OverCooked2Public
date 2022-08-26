from cgitb import grey
from concurrent.futures import thread
from enum import Enum
from multiprocessing.dummy import Process
from multiprocessing.pool import ThreadPool
from multiprocessing.spawn import prepare
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
import queue

screen = {'left': 3960, 'top': 1555, 'width': 1920, 'height': 1080}
player = Player()

que = queue.Queue()


class ComputerVision:
    boolFlag = 0
    dishesToPrepare = [[[(9999, 9999)], 'none'], [[(9999, 9999)], 'none'],
                       [[(0, 0)], 'none'], [[(0, 0)], 'none']]
    dishesToPrepare2 = {}
    platePositions = []
    count = 0
    dishText = 'none'
    utensils = {}
    ingredients = {}
    npcs = {}
    dishes = {}
    foods = {}
    playerLocation = [0, 0]
    holding = ''

    def createThreadsToFindDishes(self, screenshot, provisions):
        self.count += 1
        for i, provision in enumerate(provisions):
            self.getDishes(screenshot, provision, i)

    def getDishes(self, screenshot, provision, i):

        image = np.array(provision.image)
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image.shape
        image = cv.resize(image, (int(image.shape[1] / 4), int(image.shape[0] / 4)),
                          interpolation=cv.INTER_AREA)
        result = cv.matchTemplate(
            screenshot, image, cv.TM_CCOEFF_NORMED)

        needleW = provision.image.shape[1]
        needleH = provision.image.shape[0]

        threshold = provision.threshold
        locations = np.where(result >= threshold)
        locations = list(zip(*locations[::-1]))

        rectangles = []

        for location in locations:
            rectangle = [int(location[0]), int(
                location[1]), needleW, needleH]

            rectangles.append(rectangle)
            rectangles.append(rectangle)

        rectangles, weights = cv.groupRectangles(
            rectangles, groupThreshold=1, eps=0.5)

        points = []

        if len(rectangles):

            dishesToPrepareRanked = []

            for (x, y, w, h) in rectangles:

                centerX = x + int((w/2) / 4)
                centerY = y + int((h/2) / 4)

                points.append((centerX, centerY))

                if provision.category == "utensil":
                    self.utensils |= {
                        provision.name: self.getPlatePositions(points)}

                    cv.drawMarker(screenshot, (centerX, centerY),
                                  color=(255, 0, 255), markerType=cv.MARKER_CROSS,
                                  markerSize=10, thickness=1)

                if provision.category == "ingredient":
                    self.ingredients |= {
                        provision.name: self.getIngredient(points)}

                    cv.drawMarker(screenshot, (centerX, centerY),
                                  color=(255, 0, 255), markerType=cv.MARKER_CROSS,
                                  markerSize=10, thickness=1)

                if provision.category == "food":
                    self.foods |= {
                        provision.name: self.getIngredient(points)}

                    cv.drawMarker(screenshot, (centerX, centerY),
                                  color=(255, 0, 255), markerType=cv.MARKER_CROSS,
                                  markerSize=10, thickness=1)

                if provision.category == "dish":
                    self.dishes |= {
                        provision.name: self.getIngredient(points)}

                    cv.drawMarker(screenshot, (centerX, centerY),
                                  color=(255, 0, 255), markerType=cv.MARKER_CROSS,
                                  markerSize=10, thickness=1)

                if provision.category == "holding":
                    holding = provision.name

                    cv.drawMarker(screenshot, (centerX, centerY),
                                  color=(255, 0, 255), markerType=cv.MARKER_CROSS,
                                  markerSize=10, thickness=1)

                if threading.active_count() == 1 and self.count % 50 == 0:

                    self.createDish(self.ingredients,
                                    self.utensils, self.dishes, self.foods)

    def createDish(self, ingredients, utensils, dishes, foods):
        thread = Thread(target=player.prepareFood,
                        args=([ingredients, utensils, dishes, foods, self.playerLocation, que]))
        thread.start()

    def getPlayerLocation(self, points):
        locationx = round((points[0] - (390 / 4)) / (90 / 4)) - 1
        locationy = round((points[1] - (171 / 4)) / (90 / 4)) - 1

        return [locationy, locationx]

    def getIngredient(self, points):
        locationx = round((points[0][0] - (390 / 4)) / (90 / 4)) - 1
        locationy = round((points[0][1] - (171 / 4)) / (90 / 4)) - 1

        return [locationy, locationx]

    def getPlatePositions(self, points):
        locationx = round((points[0][0] - (390 / 4)) / (90 / 4)) - 1
        locationy = round((points[0][1] - (171 / 4)) / (90 / 4)) - 1

        return [locationy, locationx]

    def getCuttingBlockPosition(self, points):
        locationx = round((points[0][0] - (390 / 4)) / (90 / 4)) - 1
        locationy = round((points[0][1] - (171 / 4)) / (90 / 4)) - 1

        return [locationy, locationx]

    def hsvView(self, screenshot):
        hsv = cv.cvtColor(screenshot, cv.COLOR_BGR2HSV)
        lower = np.array([42, 180, 220])
        upper = np.array([120, 255, 255])
        mask = cv.inRange(hsv, lower, upper)
        result = cv.bitwise_and(
            screenshot, screenshot, mask=mask)
        indices = np.nonzero(result)
        cv.imshow('hsv', result)

        try:
            mean = int(np.mean(indices[0]))
            mean2 = int(np.mean(indices[1]))
        except:
            mean = 180
            mean2 = 220

        font = cv.FONT_HERSHEY_SIMPLEX
        org = (25, 25)
        fontScale = 1
        color = (255, 255, 255)
        thickness = 2
        markerColor = (255, 0, 255)
        markerType = cv.MARKER_CROSS

        self.playerLocation = self.getPlayerLocation([mean2, mean])
        if que.empty():
            que.put(self.playerLocation)

        cv.putText(screenshot, str(int(self.playerLocation[0])) + ',' + str(int(self.playerLocation[1])), org, font,
                   fontScale, color, thickness, cv.LINE_AA)
        cv.drawMarker(screenshot, (int(mean2), int(mean)),
                      color=markerColor, markerType=markerType,
                      markerSize=20, thickness=2)


fish = cv.imread('./foods/fish.jpg', cv.IMREAD_UNCHANGED)
seaweed = cv.imread('./foods/seaweed.jpg', cv.IMREAD_UNCHANGED)
rice = cv.imread('./foods/rice.jpg', cv.IMREAD_UNCHANGED)

fishIngredient = cv.imread('./ingredients/fish.jpg', cv.IMREAD_UNCHANGED)
seaweedIngredient = cv.imread('./ingredients/seaweed.jpg', cv.IMREAD_UNCHANGED)
riceIngredient = cv.imread('./ingredients/rice.jpg', cv.IMREAD_UNCHANGED)

cuttingBoard = cv.imread('./utensils/cuttingBoard.jpg', cv.IMREAD_UNCHANGED)
pan = cv.imread('./utensils/pan.jpg', cv.IMREAD_UNCHANGED)
plate = cv.imread('./utensils/plate.jpg', cv.IMREAD_UNCHANGED)
theExit = cv.imread('./utensils/exit.jpg', cv.IMREAD_UNCHANGED)

sushi = cv.imread('./dishes/sushi.jpg', cv.IMREAD_UNCHANGED)

holdingRice = cv.imread('./holding/rice.jpg', cv.IMREAD_UNCHANGED)

computerVision = ComputerVision()


class Provision:
    def __init__(self, image, name, category, threshold):
        self.image = image
        self.name = name
        self.category = category
        self.threshold = threshold


class Main:
    screenshot = None

    def start(self):
        with mss() as sct:
            frameCounter = 0
            provisions = [
                Provision(fishIngredient, 'fish', 'ingredient', 0.85),
                Provision(seaweedIngredient, 'seaweed', 'ingredient', 0.80),
                Provision(riceIngredient, 'rice', 'ingredient', 0.85),

                Provision(pan, 'pan', 'utensil', 0.80),
                Provision(cuttingBoard, 'cuttingBoard', 'utensil', 0.90),
                Provision(plate, 'plate', 'utensil', 0.90),
                Provision(theExit, 'exit', 'utensil', 0.80),

                Provision(sushi, 'sushi', 'dish', 0.90),

                Provision(holdingRice, 'rice', 'holding', 0.80)
            ]

            while True:
                frameCounter += 1
                self.screenshot = sct.grab(screen)
                self.screenshot = np.array(self.screenshot)
                self.screenshot = cv.resize(self.screenshot, (int(self.screenshot.shape[1] / 4), int(self.screenshot.shape[0] / 4)),
                                            interpolation=cv.INTER_AREA)
                computerVision.hsvView(self.screenshot)
                self.screenshot = cv.cvtColor(
                    self.screenshot, cv.COLOR_BGR2GRAY)
                computerVision.createThreadsToFindDishes(
                    self.screenshot, provisions)

                self.screenshot = cv.resize(self.screenshot, (1920, 1080),
                                            interpolation=cv.INTER_AREA)
                cv.imshow('Totally Awesome AI', self.screenshot)
                if cv.waitKey(1) == ord('q'):
                    cv.destroyAllWindows()
                    break


main = Main()

if __name__ == '__main__':
    main.start()
    cv.destroyAllWindows()
    print('destroyed')
