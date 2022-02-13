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
FPS = 500
fpsClock = pygame.time.Clock()

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
# 参数初始化
n_iter = 2200
sz = (n_iter,)  # size of array
t = 0.1
u = 10

Q = 0.01  # process variance
Q2 = 0.01
# z = numpy.concatenate([numpy.expand_dims(zx,axis=0),numpy.expand_dims(zy,axis=0)],axis=0)

# 分配数组空间
xhat = numpy.zeros((4, n_iter))  # a posteri estimate of x 滤波估计值
xt = numpy.zeros((4, n_iter))

P = numpy.array(((900.0, 0.0, 0.0, 0.0), (0.0, 0.01, 0.0, 0.0), (0.0, 0.0, math.pow(math.pi / 180 * 3, 2), 0.0),
                (0.0, 0.0, 0.0, 0.0)))  # a posteri error estimate滤波估计协方差矩阵
xhatminus = numpy.zeros((4, n_iter))  # a priori estimate of x 估计值
Pminus = numpy.array(((0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0),
                    (0.0, 0.0, 0.0, 0.0)))  # a prior# i error estimate估计协方差矩阵
K = numpy.array(((0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 0.0)))  # gain or blending factor卡尔曼增益
F = numpy.array(((1.0, t, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0)))
FT = numpy.transpose(F)
B = numpy.array(((0), (u), (0), (0)))
C = numpy.array(((0), (0), (t), (0)))
D = numpy.array(((0.0), (0.0), (0.0), (t)))
R = numpy.array(((900.0, 0.0, 0.0), (0.0, (math.pi / 180 * 3.0) * (math.pi / 180 * 3.0), 0.0),
                (0.0, 0.0, 0.0)))  # estimate of measurement variance, change to see effect
R2 = 900.0
H = numpy.array(((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0)))
# intial guesses
xhat[0, 0] = 400.0
xhat[1, 0] = 0.2
xhat[2, 0] = math.pi / 180 * 0.1
xhat[3, 0] = 0.2
w = numpy.random.normal(0, math.pi / 180 * 0.0, n_iter) + math.pi / 180 * 0.0
a = numpy.random.normal(0, 0.0, n_iter)
for n in range(0, 50):
    a[n] = 5
for n in range(150, 180):
    w[n] = w[n] + math.pi / 180 * (30)
for n in range(280, 310):
    w[n] = w[n] + math.pi / 180 * (30)
for n in range(410, 440):
    w[n] = w[n] + math.pi / 180 * (30)
for n in range(560, 590):
    w[n] = w[n] + math.pi / 180 * (30)
    a[n] = a[n] - 5
for n in range(810, 840):
    w[n] = w[n] + math.pi / 180 * (30)
    a[n] = a[n] + 5
for n in range(970, 1000):
    w[n] = w[n] + math.pi / 180 * (30)
for n in range(1050, 1080):
    w[n] = w[n] + math.pi / 180 * (30)
    a[n] = a[n] - 5
for n in range(1270, 1450):
    w[n] = w[n] + math.pi / 180 * (10)
for n in range(1520, 1640):
    w[n] = w[n] + math.pi / 180 * (-15)
for n in range(1700, 1760):
    w[n] = w[n] + math.pi / 180 * (-15)
for n in range(1860, 2060):
    w[n] = w[n] + math.pi / 180 * (18)

yhat = numpy.zeros((2, n_iter))  # a posteri estimate of x 滤波估计值
yt = numpy.zeros((2, n_iter))

P2 = numpy.array(((100.0, 0.0), (0.0, 0.0)))  # a posteri error estimate滤波估计协方差矩阵
yhatminus = numpy.zeros((2, n_iter))  # a priori estimate of x 估计值
Pminus2 = numpy.array(((0.0, 0.0), (0.0, 0.0)))  # a prior# i error estimate估计协方差矩阵
K2 = numpy.array(((0.0, 0.0), (0.0, 0.0)))  # gain or blending factor卡尔曼增益
F2 = numpy.array(((1.0, t), (0.0, 0.0)))
FT2 = numpy.transpose(F2)
B2 = numpy.array(((0), (u)))

# estimate of measurement variance, change to see effect
H2 = numpy.array((1.0, 0.0))
# intial guesses
yhat[0, 0] = 400.0
yhat[1, 0] = 0.2

xt[:, 0] = [400.0, 0.0, 0.0, 0.0]
yt[:, 0] = [400.0, 0.0]
for k in range(1, n_iter):
    B = numpy.array(((0), (xt[3, k - 1]), (0), (0)))
    xt[:, k] = numpy.matmul(F, xt[:, k - 1]) + B * math.cos(xt[2, k - 1]) + C * w[k - 1] + D * a[k - 1]
    B2 = numpy.array(((0), (xt[3, k - 1])))
    yt[:, k] = numpy.matmul(F2, yt[:, k - 1]) + B2 * math.sin(xt[2, k - 1])

noise = numpy.random.normal(0, 20, n_iter)
noise1 = numpy.random.normal(0, math.pi / 180 * 3.0, n_iter)
noise2 = numpy.random.normal(0, 1.0, n_iter)
noise3 = numpy.random.normal(0.0, 0.0, n_iter)
z = numpy.matmul(H, xt) + numpy.array(((noise), (noise3), (noise3)))
noise4 = numpy.random.normal(0, 10, n_iter)
z2 = numpy.matmul(H2, yt) + noise4

w = noise1 + math.pi / 180.0 * 0.1
a = noise2
for n in range(0, 50):
    a[n] = 5
for n in range(150, 180):
    w[n] = w[n] + math.pi / 180 * (30)
for n in range(280, 310):
    w[n] = w[n] + math.pi / 180 * (30)
for n in range(410, 440):
    w[n] = w[n] + math.pi / 180 * (30)
for n in range(560, 590):
    w[n] = w[n] + math.pi / 180 * (30)
    a[n] = a[n] - 5
for n in range(810, 840):
    w[n] = w[n] + math.pi / 180 * (30)
    a[n] = a[n] + 5
for n in range(970, 1000):
    w[n] = w[n] + math.pi / 180 * (30)
for n in range(1050, 1080):
    w[n] = w[n] + math.pi / 180 * (30)
    a[n] = a[n] - 5
for n in range(1270, 1450):
    w[n] = w[n] + math.pi / 180 * (10)
for n in range(1520, 1640):
    w[n] = w[n] + math.pi / 180 * (-15)
for n in range(1700, 1760):
    w[n] = w[n] + math.pi / 180 * (-15)
for n in range(1860, 2060):
    w[n] = w[n] + math.pi / 180 * (18)

k = 0

def Distance(carx, cary, centerx, centery):
    dist = math.sqrt((carx - centerx) * (carx - centerx) + (cary - centery) * (cary - centery))
    return dist

way = 0
cccc = 0
ccccR = 0

while True:
    k = k + 1
    if k == n_iter:
        break

    DISPLAYSURF.blit(raceImg, (0, 0))
    # 预测
    B = numpy.array(((0), (xhat[3, k - 1]), (0), (0)))
    zzzz = math.cos(xhat[2, k - 1])
    xhatminus[:, k] = numpy.matmul(F, numpy.transpose(xhat[:, k - 1])) + B * zzzz + C * w[k - 1] + D * a[
        k - 1]  # X(k|k-1) = AX(k-1|k-1) + BU(k) + W(k),A=1,BU(k) = 0  + B * (a + za[k-1])       F*xhat[:,k - 1]za[k-1]
    Pminus = F * P * FT + Q  # P(k|k-1) = AP(k-1|k-1)A' + Q(k) ,A=1
    # 更新
    HT = numpy.transpose(H)
    K = numpy.matmul(numpy.matmul(Pminus, HT), numpy.linalg.inv(
        numpy.matmul(H, numpy.matmul(Pminus, HT)) + R))  # Kg(k)=P(k|k-1)H'/[HP(k|k-1)H' + R],H=1
    xhat[:, k] = xhatminus[:, k] + numpy.matmul(K, (
            z[:, k] - numpy.matmul(H, xhatminus[:, k])))  # X(k|k) = X(k|k-1) + Kg(k)[Z(k) - HX(k|k-1)], H=1
    P = numpy.matmul((1 - numpy.matmul(K, H)), Pminus)  # P(k|k) = (1 - Kg(k)H)P(k|k-1), H=1

    B2 = numpy.array(((0), (xhat[3, k - 1])))
    yhatminus[:, k] = numpy.matmul(F2, numpy.transpose(yhat[:, k - 1])) + B2 * math.sin(xhat[
                                                                                            2, k - 1])  # X(k|k-1) = AX(k-1|k-1) + BU(k) + W(k),A=1,BU(k) = 0  + B * (a + za[k-1])       F*xhat[:,k - 1]za[k-1]
    Pminus2 = F2 * P2 * FT2 + Q2  # P(k|k-1) = AP(k-1|k-1)A' + Q(k) ,A=1
    # 更新
    HT2 = numpy.transpose(H2)

    K2 = numpy.matmul(Pminus2, HT2) / (
            numpy.matmul(H2, numpy.matmul(Pminus2, HT2)) + R2)  # Kg(k)=P(k|k-1)H'/[HP(k|k-1)H' + R],H=1
    yhat[:, k] = yhatminus[:, k] + K2 * (
            z2[k] - numpy.matmul(H2, yhatminus[:, k]))  # X(k|k) = X(k|k-1) + Kg(k)[Z(k) - HX(k|k-1)], H=1
    P2 = (1 - numpy.matmul(K2, H2)) * Pminus2  # P(k|k) = (1 - Kg(k)H)P(k|k-1), H=1

    way = math.tan(xhat[2, k])
    wayR = (-1) / way
    carImg2 = pygame.transform.rotate(carImg, -xhat[2, k] * 180.0 / math.pi - 90)

    height = carImg2.get_height()
    width = carImg2.get_width()
    centerx = xhat[0, k]
    centery = yhat[0, k]

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

    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
    pygame.display.update()
    fpsClock.tick(FPS)