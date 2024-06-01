import numpy as np
import matplotlib.pyplot as plt
import sympy as sympy

# ODE SOLVER
def RK4_oneStep(func, x, y, h):
    k1 = func(x, y)
    k2 = func(x + h/2, y + (h/2) * k1)
    k3 = func(x + h/2, y + (h/2) * k2)
    k4 = func(x + h, y + h * k3)
    yNext = y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
    return yNext

def RK4(func, xArr, y0, h, n):
    yArr = np.zeros((1, n))
    yArr[0,0] = y0
    for i in range(0, n-1):
        k1 = func(xArr[0, i], yArr[0, i])
        k2 = func(xArr[0, i] + h/2, yArr[0, i] + (h/2)*k1)
        k3 = func(xArr[0, i] + h/2, yArr[0, i] + (h/2)*k2)
        k4 = func(xArr[0, i] + h, yArr[0, i] + h*k3)
        yArr[0, i+1] = yArr[0, i] + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
    return yArr
# define simulation 

simulationTime = 60
dt = 0.1
size = int(simulationTime/dt)
timeArray = np.zeros((1, size))
for i in range(1, size):
    timeArray[0, i] = timeArray[0, i-1] + dt
    
# define simulating object

def vehicleModel(v, t, u, load):
    '''
    u is the gas pedal position (%pedal)
    v is the velocity
    load = passenger + cargo
    '''
    #define car base mass
    m = 500 #kg
    #define drag coefficient
    Cd = 0.24 
    #define air density
    rho = 1.225 #kg/m3
    #define cross section area
    A = 5  #m2
    #define thrust parameter
    Fp = 30 # Newton/%pedal
    #calculate acceleration
    dv_dt = (Fp/(m + load))*u - (0.5)*(rho*A*Cd/(m+load))*v**2
    return dv_dt

#initial condition 

v0 = 0 #m/s
load = 200 #kg
vArray = np.zeros((1, size))
stepSignal = np.zeros((1, size))
for i in range(size):
    if (i < 20):
        stepSignal[0, i] = 100
    elif (i < 50):
        stepSignal[0, i] = -1
    else: 
        stepSignal[0, i] = 20
        
# simulation for finding time k and T

# vArray = RK4(lambda t, v: vehicleModel(v, t, stepSignal[0, int(t/dt)], load), timeArray, v0, dt, size)

# define PID controller
# controller parametres
Kc = 1
Ti = 0.5
Td = 10

integral = 0
previousError = 0
sumIntegral = 0
uBias = 0
integralError = np.zeros((1, size))
errorArray = np.zeros((1, size))

control_noise_std = 1.2 
#
for i in range(0, size - 1):
    error = stepSignal[0, i] - vArray[0, i]
    P = Kc * error
    # Integral term
    integralError[0, i+1] = integralError[0, i] + error*dt
    I = (Kc / Ti) * integralError[0, i+1]
    # Derivative term
    D = (Kc*Td*(error-previousError))/dt
    previous_error = error
    # PID output
    u = P + I + D
    u += np.random.normal(0, control_noise_std)
    vArray[0, i+1] = RK4_oneStep(lambda t,v: vehicleModel(v, t, u, load), timeArray[0, i], vArray[0,i], dt)
    


# data processing and plotting
plt.plot(timeArray[0, :], stepSignal[0, :])
plt.plot(timeArray[0, :], vArray[0, :])
plt.show()