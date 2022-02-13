import sys
import os
import time
import pygame
from pygame.locals import *
import numpy
import pylab
import random
import math

pygame.init()
# FPS = 500
# fpsClock = pygame.time.Clock()

# 화면 타이틀 설정
pygame.display.set_caption("racegame") #게임 이름

# FPS
clock = pygame.time.Clock()

DISPLAYSURF = pygame.display.set_mode((1006, 1006))
pygame.display.set_caption('car race')

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

raceImg = pygame.image.load('race3.png')
carImg = pygame.image.load('car.png')

racex = 5
racey = 5
carx = 500
cary = 500
raceImg = pygame.transform.scale(raceImg, (1001, 1001))
carImg = pygame.transform.scale(carImg, (50, 50))
carImg2 = pygame.transform.rotate(carImg, 0)

DISPLAYSURF.fill(WHITE)

i = 0
j = 0
j2 = 0
vx = 0
vy = 0
way = 0
wayR = 0

def Distance(carx, cary, centerx, centery):
    dist = math.sqrt((carx - centerx) * (carx - centerx) + (cary - centery) * (cary - centery))
    return dist

DISPLAYSURF.fill(WHITE)

DISPLAYSURF.blit(raceImg, (0,0))
Space = numpy.zeros([1005, 1005])
pixObj = pygame.PixelArray(DISPLAYSURF)
for xx in range(0, 1000):
    for yy in range(0, 1000):
        gotColor = DISPLAYSURF.get_at((xx, yy))
        #print(xx, yy , gotColor)
        if sum(gotColor)<950:

            # for xxx in range(5):
            # for yyy in range(5):
            # Space[5*xx+xxx][5*yy+yyy] = 1
            Space[xx-1:xx+1,yy-1:yy+1] = 1


# 这里是假设A=1，H=1的情况
del pixObj

k = 0

def Distance(carx, cary, centerx, centery):
    dist = math.sqrt((carx - centerx) * (carx - centerx) + (cary - centery) * (cary - centery))
    return dist

way = 0
cccc = 0
ccccR = 0

centerx = 500
centery = 500
car_speed_x = 0
car_speed_y = 0

while True:
    dt = clock.tick(30)


    DISPLAYSURF.blit(raceImg, (0, 0))
    for event in pygame.event.get(): # 어떤 이벤트가 발생하였는가?
        if event.type == pygame.QUIT: # 창이 닫히는 이벤트가 발생하였는가?
            running = False

        if event.type == pygame.KEYDOWN: # 키가 눌러졌는지 확인
            if event.key == pygame.K_LEFT: # 캐릭터를 왼쪽으로
                car_speed_x = -1
                way = 180.0 / math.pi
                wayR = (-1) / way
                carImg2 = pygame.transform.rotate(carImg, 90)
            elif event.key == pygame.K_RIGHT:
                car_speed_x = 1
                way = 180.0 / math.pi
                wayR = (-1) / way
                carImg2 = pygame.transform.rotate(carImg, 270)
            elif event.key == pygame.K_UP:
                car_speed_y = -1
                way = 180.0 / math.pi
                wayR = (-1) / way
                carImg2 = pygame.transform.rotate(carImg, 0)
            elif event.key == pygame.K_DOWN:
                car_speed_y = 1
                way = 180.0 / math.pi
                wayR = (-1) / way
                carImg2 = pygame.transform.rotate(carImg, 180)
            elif event.key == pygame.K_q:
                pygame.quit()
                quit()
                
            print("pressed")
        if event.type == pygame.KEYUP: # 방향키를 떼면 멈춤
            if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                car_speed_x = 0
            elif event.key == pygame.K_UP or event.key == pygame.K_DOWN:
                car_speed_y = 0
            
    centerx += car_speed_x
    centery += car_speed_y

    height = carImg2.get_height()
    width = carImg2.get_width()


    carx = centerx - width / 2
    cary = centery - height / 2
    dist = Distance(carx, cary, centerx, centery)
    DISPLAYSURF.blit(carImg2, (int((centerx) - (height) / 2), int((centery) - (width) / 2)))
    # print(k, (centerx), (centery), (way),(wayR) )
    cccc = ((centery) - ((way) * (centerx)))
    ccccR = ((centery) - ((wayR) * (centerx)))

    for x in range(1, 200):
        dist = Distance(centerx + x, way * (centerx + x) + cccc, centerx, centery)
        if dist >= 25:

            comx1 = centerx + x
            comy1 = way * (centerx + x) + cccc
            comx2 = centerx - x
            comy2 = way * (centerx - x) + cccc
            if x == 1:
                comx1 = centerx
                comy1 = centery + 25
                comx2 = centerx
                comy2 = centery - 25
            fcx = x  # 从车模中心到车模移动方向的未端的向量(fcx,fcy)
            fcy = comy1 - centery
            break

    for x in range(1, 200):
        dist = Distance(centerx + x, wayR * (centerx + x) + ccccR, centerx, centery)
        if dist >= 25:
            comx3 = centerx + x
            comy3 = wayR * (centerx + x) + ccccR
            comx4 = centerx - x
            comy4 = wayR * (centerx - x) + ccccR
            if x == 1:
                comx3 = centerx
                comy3 = centery - 25
                comx4 = centerx
                comy4 = centery + 25
            fcx2 = x  # 从车模中心到车模移动垂直方向的未端的向量(fcx2,fcy2)
            fcy2 = comy3 - centery
            break

    comx1 = centerx + fcx + fcx2
    comy1 = centery + fcy + fcy2
    comx2 = centerx + fcx - fcx2
    comy2 = centery + fcy - fcy2
    comx3 = centerx - fcx - fcx2
    comy3 = centery - fcy - fcy2
    comx4 = centerx - fcx + fcx2
    comy4 = centery - fcy + fcy2

    if ( comx3 < 0 or comx3 >= 1005 or comy3 < 0 or comy3 >= 1005 or comx4 < 0 or comx4 >= 1005 or comy4 < 0 or comy4 >= 1005 or comx1 < 0 or comx1 >= 1005 or
            comy1 < 0 or comy1 >= 1005 or comx2 < 0 or comx2 >= 1005 or comy2 < 0 or comy2 >= 1005):
        continue

    pixObj = pygame.PixelArray(DISPLAYSURF)
    pixObj[int(centerx) - 2][int(centery)] = RED
    pixObj[int(centerx) - 1][int(centery)] = RED
    pixObj[int(centerx)][int(centery)] = RED
    pixObj[int(centerx) + 1][int(centery)] = RED
    pixObj[int(centerx) + 2][int(centery)] = RED
    pixObj[int(centerx)][int(centery) - 2] = RED
    pixObj[int(centerx)][int(centery) - 1] = RED
    pixObj[int(centerx)][int(centery) + 1] = RED
    pixObj[int(centerx)][int(centery) + 2] = RED

    if (Space[int(comx1)][int(comy1)] == 1) or (Space[int(comx2)][int(comy2)] == 1) or (
            Space[int(comx3)][int(comy3)] == 1) or (Space[int(comx4)][int(comy4)] == 1):
        pygame.draw.line(DISPLAYSURF, RED, (50, 50), (995, 995), 20)
        pygame.draw.line(DISPLAYSURF, RED, (995, 50), (50, 995), 20)

        i = i + 1

    del pixObj

    pygame.display.update()