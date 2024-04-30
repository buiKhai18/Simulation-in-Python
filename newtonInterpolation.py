import numpy as np 
import csv
import sympy as sp
import matplotlib.pyplot as plt

def importData(filePath):
    
    # import data from csv file
    with open(filePath, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
    inputArr = np.array([float(col[0]) for col in data[1:]])
    outputArr = np.array([float(col[1]) for col in data[1:]])
    inputLabel = data[0][0]
    outputLabel = data[0][1]
    # print(inputArr)
    # print(np.size(inputArr))
    # print(outputArr)
    # print(np.size(outputArr))
    return inputLabel, outputLabel, inputArr, outputArr

def newtonInterpolation(scheme, filePath):
    
    #This section is used for initialize data and 
    #the difference matrix from the imported data
    xLabel, yLabel, xData, yData  = importData(filePath);
    matSize = np.size(xData)
    differenceMatrix = np.zeros((matSize, matSize))
    differenceMatrix[:, 0] = yData
    for j in range(1, matSize):
        for i in range(matSize - j):
            differenceMatrix[i, j] = (differenceMatrix[i+1, j-1] - differenceMatrix[i, j-1]) / (xData[i+j] - xData[i])
    
    # This section is for the Newton Forward interpolation
    
    x = sp.symbols('x')
    if (scheme == 'fwd'): 
        func = differenceMatrix[0,0]
        for i in range(1, matSize):
            terms = 1
            for j in range(i):
                terms *= (x-xData[j])
            func += differenceMatrix[0, i]*terms
        # print(func)
    elif(scheme == 'bwd'):
        func = differenceMatrix[matSize - 1, 0]
        for i in range(1,matSize):
            terms = 1
            for j in range(i):
                terms *= (x-xData[matSize - j - 1])
            func += differenceMatrix[matSize-i-1,i]*terms
         
    interpolatedFunction = sp.lambdify(x, func)
    xValues = np.linspace(min(xData), max(xData), 100)
    yValues = interpolatedFunction(xValues)

    #Data plotting
    
    plt.plot(xValues, yValues, label = "Interpolated Function")
    plt.scatter(xData, yData, color = 'red', label = 'Data Points')
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title("Newton Interpolation")
    plt.legend(loc = 'best')
    plt.show()

#newtonInterpolation('fwd',r"C:\Users\Admin\Desktop\VSCode\Python\Luyen tap\ODE_Solver\testInterpolation.csv");