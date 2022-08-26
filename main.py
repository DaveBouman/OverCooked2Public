import enum
from math import prod
from multiprocessing.connection import wait
from re import T
from threading import Thread
import cv2 as cv
import numpy as np
import pyautogui
import time


class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position


def returnPath(currentNode, grid):
    path = []
    rows, columns = np.shape(grid)
    # here we create the initialized result grid with -1 in every position
    result = [[-1 for i in range(columns)] for j in range(rows)]
    current = currentNode
    while current is not None:
        path.append(current.position)
        current = current.parent
    # Return reversed path as we need to show from start to end path
    path.reverse()
    startValue = 0
    # we update the path of start to end found by A-star serch with every step incremented by 1
    for i in range(len(path)):
        result[path[i][0]][path[i][1]] = startValue
        startValue += 1

    movements = []
    for i in range(len(path)):
        if i + 1 < len(path):
            if path[i][0] < path[i + 1][0]:
                movements.append([1, 's'])
                continue

            if path[i][0] > path[i + 1][0]:
                movements.append([2, 'w'])
                continue

            if path[i][1] > path[i + 1][1]:
                movements.append([3, 'a'])
                continue

            if path[i][1] < path[i + 1][1]:
                movements.append([4, 'd'])
                continue
    return result, movements


def search(grid, cost, start, end):
    # Create start and end node with initized values for g, h and f
    startNode = Node(None, tuple(start))
    # startNode.g = startNode.h = startNode.f = 0
    endNode = Node(None, tuple(end))
    # endNode.g = endNode.h = endNode.f = 0

    grid[endNode.position[0]][endNode.position[1]] = 0
    # Initialize both yet_to_visit and visited list
    # in this list we will put all node that are yet_to_visit for exploration.
    # From here we will find the lowest cost node to expand next
    nodesToVisit = []
    # in this list we will put all node those already explored so that we don't explore it again
    visitedNodes = []

    # Add the start node
    nodesToVisit.append(startNode)

    # Adding a stop condition. This is to avoid any infinite loop and stop
    # execution after some reasonable number of steps
    outerIterations = 0
    maxIterations = (len(grid) // 2) ** 10

    # what squares do we search . serarch movement is left-right-top-bottom
    # (4 movements) from every positon

    move = [[-1, 0],  # go up
            [0, -1],  # go left
            [1, 0],  # go down
            [0, 1]]  # go right

    # find grid has got how many rows and columns
    rows, columns = np.shape(grid)

    # Loop until you find the end

    while len(nodesToVisit) > 0:

        # Every time any node is referred from yet_to_visit list, counter of limit operation incremented
        outerIterations += 1

        # Get the current node
        currentNode = nodesToVisit[0]
        currentIndex = 0
        for index, item in enumerate(nodesToVisit):
            if item.f < currentNode.f:
                currentNode = item
                currentIndex = index

        # if we hit this point return the path such as it may be no solution or
        # computation cost is too high
        if outerIterations > maxIterations:
            print("giving up on pathfinding too many iterations")
            return returnPath(currentNode, grid)

        # Pop current node out off yet_to_visit list, add to visited list
        nodesToVisit.pop(currentIndex)
        visitedNodes.append(currentNode)

        # test if goal is reached or not, if yes then return the path
        if currentNode == endNode:
            grid[endNode.position[0]][endNode.position[1]] = 1
            return returnPath(currentNode, grid)

        # Generate children from all adjacent squares
        children = []

        for newPosition in move:

            # Get node position
            nodePosition = (
                currentNode.position[0] + newPosition[0], currentNode.position[1] + newPosition[1])

            # Make sure within range (check if within grid boundary)
            if (nodePosition[0] > (rows - 1) or
                nodePosition[0] < 0 or
                nodePosition[1] > (columns - 1) or
                    nodePosition[1] < 0):
                continue

            # Make sure walkable terrain
            if grid[nodePosition[0]][nodePosition[1]] != 0:
                continue

            # Create new node
            newNode = Node(currentNode, nodePosition)

            # Append
            children.append(newNode)

        # Loop through children
        for child in children:

            # Child is on the visited list (search entire visited list)
            if len([visitedChild for visitedChild in visitedNodes if visitedChild == child]) > 0:
                continue

            # Create the f, g, and h values
            child.g = currentNode.g + cost
            # Heuristic costs calculated here, this is using eucledian distance
            child.h = (((child.position[0] - endNode.position[0]) ** 2) +
                       ((child.position[1] - endNode.position[1]) ** 2))

            child.f = child.g + child.h

            # Child is already in the yet_to_visit list and g cost is already lower
            if len([i for i in nodesToVisit if child == i and child.g > i.g]) > 0:
                continue

            # Add the child to the yet_to_visit list
            nodesToVisit.append(child)


class Player(Thread):
    cost = 1
    grid = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

    def pressMovementKeys(self, directions):
        for direction in directions:
            pyautogui.keyDown(direction[1])
            time.sleep(0.0899)
            pyautogui.keyUp(direction[1])

    def pressMovementKey(self, direction):
        pyautogui.keyDown(direction)
        time.sleep(0.0928)
        pyautogui.keyUp(direction)

    def GrabPlaceItem(self):
        pyautogui.keyDown('space')
        pyautogui.keyUp('space')

    def ChangePlayer(self):
        pyautogui.keyDown('shift')
        pyautogui.keyUp('shift')

    def cutItem(self):
        pyautogui.keyDown('space')
        pyautogui.keyUp('space')
        pyautogui.keyDown('ctrl')
        time.sleep(7)
        pyautogui.keyUp('ctrl')

    def dash(self):
        pyautogui.keyDown('altleft')
        pyautogui.keyUp('altleft')

    def switchPlayer(self):
        pyautogui.keyDown('shift')
        pyautogui.keyUp('shift')

    def goToStart(self):
        pyautogui.keyDown("w")
        pyautogui.keyDown("a")
        time.sleep(8)
        pyautogui.keyUp('w')
        pyautogui.keyUp('a')

    def goTo(self, start, end):
        path = search(self.grid, self.cost, start, end)
        if path[1]:
            self.pressMovementKeys(path[1])
            lastIndex = [index[-1] for index in path]
            if lastIndex[1][0] == 1:  # S
                end[0] = end[0] - 1
            elif lastIndex[1][0] == 2:  # W
                end[0] = end[0] + 1
            elif lastIndex[1][0] == 3:  # A
                end[1] = end[1] + 1
            elif lastIndex[1][0] == 4:  # D
                end[1] = end[1] - 1
        return end

    def prepareFish(self, plateLocation, productLocation, choppingBlockLocation):
        start = [0, 1]
        self.goTo(start, productLocation)
        self.GrabPlaceItem()

        location = self.goTo(productLocation, choppingBlockLocation)
        self.cutItem()
        self.GrabPlaceItem()

        location = self.goTo(location, plateLocation)
        self.GrabPlaceItem()
        self.GrabPlaceItem()

        location = self.goTo(location, [2, 12])
        self.GrabPlaceItem()
        self.goToStart()

    def checkLocation(self, que, goToObject, message):
        playerLocation = que.get()
        playerLocation = que.get()
        playerLocation = que.get()
        location = self.goTo(playerLocation, goToObject)
        playerLocation = que.get()
        playerLocation = que.get()
        playerLocation = que.get()
        while playerLocation[0] is not location[0] and playerLocation[1] is not location[1]:
            playerLocation = que.get()
            playerLocation = que.get()
            playerLocation = que.get()
            location = self.goTo(playerLocation, goToObject)
            playerLocation = que.get()
            playerLocation = que.get()
            playerLocation = que.get()
            playerLocation = que.get()
        time.sleep(0.1)
        self.GrabPlaceItem()
        print(message)

    def prepareFood(self, ingredients, utensils, dishes, foods, playerLocation, que):
        self.checkLocation(que, ingredients['rice'], 'rice')
        self.checkLocation(que,  utensils['pan'], 'pan')
        self.checkLocation(que, ingredients['seaweed'], 'seaweed')
        self.checkLocation(que, utensils['plate'], 'plate')
        self.checkLocation(que,  utensils['pan'], 'pan')
        self.checkLocation(que, utensils['plate'], 'plate')
