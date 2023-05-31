# Load basic libraries.
from skimage.io import imread, imshow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2 
from PIL import Image
import csv
import copy, math
from lab_utils_logistic import sigmoid


def compute_cost_logistic_reg(X, y, w, b, lambda_ = 1):
    """
    Computes the cost over all examples
    Args:
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
    Returns:
      total_cost (scalar):  cost 
    """

    m,n  = X.shape
    cost = 0.
    for i in range(m):
        z_i = np.dot(X[i], w) + b                                      #(n,)(n,)=scalar, see np.dot
        f_wb_i = sigmoid(z_i)                                          #scalar
        aux = np.where((1-f_wb_i) > 0.0000000001, (1-f_wb_i), -10)
        aux2 = np.where((f_wb_i) > 0.0000000001, (f_wb_i), -10)
        cost +=  -y[i]*np.log(aux2, out=aux2, where=aux2>0) - (1-y[i])*np.log(aux, out=aux, where=aux>0)      #scalar
        #cost +=  -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)      #scalar
             
    cost = cost/m                                                      #scalar

    reg_cost = 0
    for j in range(n):
        reg_cost += (w[j]**2)                                          #scalar
    reg_cost = (lambda_/(2*m)) * reg_cost                              #scalar
    
    total_cost = cost + reg_cost                                       #scalar
    return total_cost    

def compute_gradient_logistic_reg(X, y, w, b, lambda_): 
    """
    Computes the gradient for linear regression 
 
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
    Returns
      dj_dw (ndarray Shape (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar)            : The gradient of the cost w.r.t. the parameter b. 
    """
    m,n = X.shape
    dj_dw = np.zeros((n,))                            #(n,)
    dj_db = 0.0                                       #scalar

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i],w) + b)          #(n,)(n,)=scalar
        err_i  = f_wb_i  - y[i]                       #scalar
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i,j]      #scalar
        dj_db = dj_db + err_i
    dj_dw = dj_dw/m                                   #(n,)
    dj_db = dj_db/m                                   #scalar

    for j in range(n):
        dj_dw[j] = dj_dw[j] + (lambda_/m) * w[j]

    return dj_db, dj_dw  

def gradient_descent(X, y, w_in, b_in, alpha, r_lambda, num_iters): 
    """
    Performs batch gradient descent
    
    Args:
      X (ndarray (m,n)   : Data, m examples with n features
      y (ndarray (m,))   : target values
      w_in (ndarray (n,)): Initial values of model parameters  
      b_in (scalar)      : Initial values of model parameter
      alpha (float)      : Learning rate
      r_lambda (float)     : Regularization rate
      num_iters (scalar) : number of iterations to run gradient descent
      
    Returns:
      w (ndarray (n,))   : Updated values of parameters
      b (scalar)         : Updated value of parameter 
    """
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters+1):
        # Calculate the gradient and update the parameters
        dj_db, dj_dw = compute_gradient_logistic_reg(X, y, w, b, r_lambda)   

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               
      
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( compute_cost_logistic_reg(X, y, w, b, r_lambda) )

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")
        
    return w, b, J_history

def y_change(y, cl):
    """
    Creates an independent y vector that only holds 1's for
    the selected class and zero for the rest
    
    Args:
      y (ndarray (m,)) : target values
      cl (scalar)      : The class we are studying.
      
    Returns:
      y_pr (ndarray (n,))   : Array holding only 1's for the 
                              analyzed class.
    """
    y_pr=[]
    for i in range(0, len(y)):
        if y[i] == cl:
            y_pr.append(1)
        else:
            y_pr.append(0)
    return y_pr

def find_param(X, y):
    """
    Creates the w_i vector for the given class.
    
    Args:
      X (ndarray (m,n)    : Data, m examples with n features
      y (ndarray (m,))    : Target values
      
    Returns:
      theta_list (ndarray (n,)) : This is a matrix that will hold a row for the w values
                                  for every i class. 
    """

    alph = 0.1
    r_lambda = 0.7
    iters = 1000

    y_uniq = list(set(y.flatten()))
    theta_list = []
    for i in y_uniq:
        w_in = np.random.rand(X.shape[1])
        b_in = 0.5
        y_tr = pd.Series(y_change(y, i))
        # y_tr = y_tr[:, np.newaxis]
        np.array(y_tr)[:, np.newaxis]
        print(f"\n\nWe will find the weights for class: {i}")
        theta1, _ , _ = gradient_descent(X, y_tr, w_in, b_in, alph, r_lambda, iters)
        theta_list.append(theta1)
    return theta_list 

def predict(theta_list, X, y):
    y_uniq = list(set(y.flatten()))
    y_hat = [0]*len(y)
    for i in range(0, len(y_uniq)):
        y_tr = y_change(y, y_uniq[i])
        # y1 = sigmoid(x, theta_list[i])
        y1 = sigmoid(np.dot(X, theta_list[i]))
        for k in range(0, len(y)):
            if y_tr[k] == 1 and y1[k] >= 0.5:
                y_hat[k] = y_uniq[i]
    return y_hat

def featureVec(image):
    featureVec = []
    #featureVec.append(featureColor(image).flatten())
    featureVec.append(featureRGBAverage(image).flatten())
    featureVec.append(featureRedHistogram(image).flatten())
    featureVec.append(featureGreenHistogram(image).flatten())
    featureVec.append(featureBlueHistogram(image).flatten())

    #featureVecArray = np.concatenate((featureVec[0],featureVec[1],featureVec[2],featureVec[3],featureVec[4] ))
    featureVecArray = np.concatenate((featureVec[0],featureVec[1],featureVec[2],featureVec[3] ))
    
    return featureVecArray

def featureColor(image):
    featureColor = np.zeros((int(image.shape[0]),int(image.shape[1])))
    for i in range(0,image.shape[0]):
        for j in range(0,image.shape[1]):
            featureColor[i][j] = ((int(image[i,j,0]) + int(image[i,j,1]) + int(image[i,j,2]))/3)
    return featureColor.reshape(int(image.shape[0]),int(image.shape[1]))

def featureRGBAverage(image):
    featureRed = 0
    featureGreen = 0
    featureBlue = 0
    for i in range(0,image.shape[0]):
        for j in range(0,image.shape[1]):
            featureRed += int(image[i,j,0])
            featureGreen += int(image[i,j,1])
            featureBlue += int(image[i,j,2])
    featureRed = featureRed / (int(image.shape[0])*int(image.shape[1]))
    featureGreen = featureGreen / (int(image.shape[0])*int(image.shape[1]))
    featureBlue = featureBlue / (int(image.shape[0])*int(image.shape[1]))
    featureRGBAverage = np.array([featureRed,featureGreen,featureBlue])
    return featureRGBAverage

def featureRedHistogram(image):
    featureHistogram = np.zeros(256)
    for i in range(0,image.shape[0]):
        for j in range(0,image.shape[1]):
            featureHistogram[int(image[i,j,0])]+=1
    return featureHistogram

def featureGreenHistogram(image):
    featureHistogram = np.zeros(256)
    for i in range(0,image.shape[0]):
        for j in range(0,image.shape[1]):
            featureHistogram[int(image[i,j,1])]+=1
    return featureHistogram

def featureBlueHistogram(image):
    featureHistogram = np.zeros(256)
    for i in range(0,image.shape[0]):
        for j in range(0,image.shape[1]):
            featureHistogram[int(image[i,j,2])]+=1
    return featureHistogram

def showImage(image) :  
    # If running local
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # If running in google
    # cv2_imshow(img_load)

# -------------------------------------------------------------------------------------- Main

'''
X = []
Y = []

with open("pictures.csv", "r") as f:
    reader = csv.reader(f, delimiter="\t")
    for i, line in enumerate(reader):
        if(line[0].split(",")[0]=="Route"):
            continue
        route = line[0].split(",")[0]
        Y.append(line[0].split(",")[1])
        image = cv2.imread(route, cv2.IMREAD_COLOR)
        nArray = featureVec(image)
        X.append(nArray)

with open("features.csv", "w",newline='') as csvfile:
    writer = csv.writer(csvfile)
    for fea in X:
        writer.writerow(fea)
'''
'''
X = []
Y = []
with open("pictures.csv", "r") as f1:
    reader = csv.reader(f1, delimiter="\t")
    for i, line in enumerate(reader):
        if(line[0].split(",")[0]=="Route"):
            continue
        tag = line[0].split(",")[1]
        if tag == 'Shrek':
            Y.append(1)
        elif tag == 'DemonSlayer':
            Y.append(2)
        elif tag == 'TheSimpsons':
            Y.append(3)

with open("features.csv", "r") as f2:
    reader = csv.reader(f2, delimiter="\t")
    for i, line in enumerate(reader):
        feat = line[0].split(',')
        nArray = np.array(feat,dtype=float)
        X.append(nArray)

X = np.array(X).flatten().reshape(1200,771) 
Y = np.array(Y).flatten()

print(X.shape,Y.shape)
theta_list = find_param(X, Y)

print(theta_list)
with open("thetaList.csv", "w",newline='') as csvfile:
    writer = csv.writer(csvfile)
    for theta in theta_list:
        writer.writerow(theta)
'''
theta_list = []
#Plotting the actual and predicted values
with open("./model/thetaList.csv", "r") as f2:
    reader = csv.reader(f2, delimiter="\t")
    for i, line in enumerate(reader):
        feat = line[0].split(',')
        nArray = np.array(feat,dtype=float)
        theta_list.append(nArray)

X_Image = []
Y_image = [1,2,3]

with open("./outputs/testImages.csv", "r") as f:
    reader = csv.reader(f, delimiter="\t")
    for i, line in enumerate(reader):
        if(line[0].split(",")[0]=="Route"):
            continue
        route = line[0].split(",")[0]
        image = cv2.imread(route, cv2.IMREAD_COLOR)
        nArray = featureVec(image)
        X_Image.append(nArray)

X_Image = np.array(X_Image).flatten().reshape(3,771) 
Y_image = np.array(Y_image).flatten()

y_hat = predict(theta_list, X_Image, Y_image)

if(y_hat[0]==1 and y_hat[1]==0 and y_hat[2]==0):
    print("Shrek")
elif(y_hat[0]==0 and y_hat[1]==2 and y_hat[2]==0):
    print("Demon Slayer")
elif(y_hat[0]==0 and y_hat[1]==0 and y_hat[2]==3):
    print("The Simpsons")
else:
    print("Inconclusive, please pick another image")

'''
f1 = plt.figure()
c = [i for i in range (1,len(Y)+1,1)]
plt.plot(c,Y,color='r',linestyle='-')
plt.plot(c,y_hat,color='b',linestyle='-')
plt.xlabel('Value')
plt.ylabel('Class')
plt.title('Actual vs. Predicted')
plt.show()

#Plotting the error
f1 = plt.figure()
c = [i for i in range(1,len(Y)+1,1)]
plt.plot(c,Y-y_hat,color='green',linestyle='-')
plt.xlabel('index')
plt.ylabel('Error')
plt.title('Error Value')
plt.show()


header = ["Feature Color", "RGB Average" , "Red Histogram", "Green Histogram", "Blue Histogram"]
with open('features2.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    #writer.writerow(header)
    for feature in res:
        #writer.writerow(feature)
        print(feature)
        break
'''