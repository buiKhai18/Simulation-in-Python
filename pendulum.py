import numpy as np
import matplotlib.pyplot as plt
import os

"""
    In this code, I did not use any numerical method, it just bases on
    physics and the aim is to test my skills on basical modelling in Python
"""
os.system("cls")
g = 9.81
deltaT = 0.01 #Sample extracting time
l = -1
N = -1
def drawPendulum(x, y):
    plt.axis("equal")
    plt.plot(x, y, color = 'red', marker = 'o', markersize = 20)
    plt.plot([0, x], [0, y], linestyle = '-', linewidth = 3)

def init_tArr(N):
    tArr = np.zeros((1, N))
    for i in range(1, N):
        tArr[0,i] = tArr[0,0] + i*deltaT
    return tArr

def simulate(xArr, yArr):
    fig, ax = plt.subplots()
    ax.set_xlim(-30,30)
    ax.set_ylim(-30,30)
    for i in range(N):
        plt.clf()
        drawPendulum(xArr[0, i], yArr[0, i])
        if (i % 5) == 1:
            plt.pause(0.01)

def initAngleForNPeriod(thetaBase, N):
    theta = np.zeros((1,N))
    count = 2
    for i in range(len(thetaBase)):
        theta[0, i] = thetaBase[i]
    for i in range(len(thetaBase)+1, N):
        if(i == count*len(thetaBase)):
            count+=1
        else:
            theta[0, i] = -theta[0, i - len(thetaBase)]     
    return theta

while (l <= 0) or (N <= 0):
    l = float(input("Length of string: "))
    N = int(input("Number of samples: "))
    theta0 = float(input("The initial angle (degree): "))

T = 2*np.pi*np.sqrt(g/l)
tArr = init_tArr(N)

theta0Rad = np.deg2rad(theta0)
theta = np.linspace(-theta0Rad, theta0Rad, int(T/deltaT))

countPeriod = 2
thetaBase = theta

theta = initAngleForNPeriod(thetaBase, N)

xArr = l*np.sin(theta)
yArr = -l*np.cos(theta)
plt.show(block=False) 
simulate(xArr, yArr)


    




