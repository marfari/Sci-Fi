import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math as m
from scipy.optimize import minimize

df1 = pd.read_csv('4TSEPLFIM00471/b659f4d6-bbd2-4a74-8362-e948ec67fc81/mat_results/merged_fibres_all_um_4TSEPLFIM00471_SiPM_A.csv',sep="\t",header=None) # fibers
df2 = pd.read_csv('ClearFiber1-L-P2/86fc5817-4dbb-4ddf-a38d-01c8bcac0206/mat_results/merged_fibres_all_um_ClearFiber1-L-P2_SiPM_A.csv',sep="\t",header=None) # clear fibers
pins2 = pd.read_csv('ClearFiber1-L-P2/86fc5817-4dbb-4ddf-a38d-01c8bcac0206/mat_results/pins_all_um_ClearFiber1-L-P2_SiPM_A.csv', sep="\t",header=None)

df1.columns = ["fiber","x", "y", "r", "smth","smth1","smth2","smth3","smth4","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0"]
df2.columns = ["fiber","x", "y", "r", "smth","smth1","smth2","smth3","smth4","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0"]
df1.head()
df2.head()

df2.drop(index=[2849,2850,2851,2852,2853,2854,2855,2856,2857,2858,2859,2860],inplace=True)
pins2.columns = ["x", "y", "r", "smth","smth1","smth2","smth3","smth4"]
pins2.head()

#print(df2)

def find_intersections(x1, y1, x2, y2, r1, r2):
    """ Finds intersections between circle 1: (x1,y1) radius r1; with circle 2: (x2,y2) radius r2 """
    d = m.sqrt((x2-x1)**2 + (y2-y1)**2) # distance between the centres of the 2 circles
    
    if d > r1 + r2: # circles do not intersect
        #print("Circles do not intersect")
        return None 
    if d < abs(r2-r1): # one circle is within the other
        #print("One circle is within the other")
        return False
    if d == 0 and r1 == r2: # two circles coincide
        #print("Two circles coincide")
        return True
    else: # circles intersect
        a = (r1**2 - r2**2 + d**2)/(2*d)
        h = m.sqrt(r1**2 - a**2)
        x3 = x1 + a*(x2 - x1)/d   
        y3 = y1 + a*(y2 - y1)/d   
        xP1 = x3 + h*(y2 - y1)/d     
        yP1 = y3 - h*(x2 - x1)/d 

        xP2 = x3 - h*(y2 - y1)/d
        yP2 = y3 + h*(x2 - x1)/d
        #print("Circles intersect in P1=(%f,%f) and P2=(%f,%f)",xP1,yP1,xP2,yP2)
        return (xP1, yP1, xP2, yP2)
    
def translate(x, y, mark_x1, X, Y):
    return (x - (X - mark_x1), y - Y)

def rotate(x, y, theta):
    return (m.cos(theta)*x - m.sin(theta)*y, m.sin(theta)*x + m.cos(theta)*y)
    
def transform(x, y, mark_x1, X, Y, theta):
    """ Makes transformation to new position of the alignements pins. x1 is the x-coordinate of the 1st mark. """
    return rotate( translate(x, y, mark_x1, X, Y)[0], translate(x, y, mark_x1, X, Y)[1], theta)
    
# df2.drop(index=[2849,2850,2851,2852,2853,2854,2855,2856,2857,2858,2859,2860],inplace=True)
    
def overlap_area(df1, df2, pins2, X, Y, theta):
    """ Calculates the overlap area between fiber mat 1 (df1) and reflected fiber mat 2 (df2) """
    df1.columns = ["fiber","x", "y", "r", "smth","smth1","smth2","smth3","smth4","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0"]
    df2.columns = ["fiber","x", "y", "r", "smth","smth1","smth2","smth3","smth4","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0"]
    df1.head()
    df2.head()

    x1 = df1['x']
    x2 = df2['x']
    y1 = df1['y']
    y2 = df2['y']
    r1 = df1['r']
    r2 = df2['r']
    
    pins2.columns = ["x", "y", "r", "smth","smth1","smth2","smth3","smth4"]
    pins2.head()
    mark_x1 = pins2['x'].iloc[0]

    if(len(df1) < len(df2)): L = len(df1)
    else: L = len(df2)

    # Compute overlap area
    A_overlap = 0
    A_tot1 = 0
    A_tot2 = 0
    for i in range(L):

        # Transform 2nd mat to a new reference frame, where we will put the alignment pins
        (xp2, yp2) = transform(x2.iloc[-i], y2.iloc[-i], mark_x1, X, Y, theta)     

        # Find the intersections between the 2 mats
        intersections = find_intersections(x1.iloc[i], y1.iloc[i], xp2, yp2, r1.iloc[i], r2.iloc[i]) # xp2, yp2
        
        A1 = m.pi*(r1.iloc[i])**2
        A2 = m.pi*(r2.iloc[-i])**2
        
        A_tot1 += A1
        A_tot2 += A2
        
        if intersections == None:
            continue
        elif intersections == False:
            if(A1 < A2):
                A_overlap += A1
            else:
                A_overlap += A2
        elif intersections == True:
            A_overlap += A1 # A1=A2
        else:
            xP1 = intersections[0]
            yP1 = intersections[1]
            xP2 = intersections[2]
            yP2 = intersections[3]
            
            P1_P2 = m.sqrt((xP2 - xP1)**2 + (yP2 - yP1)**2)
            A_P1 = m.sqrt((xP1 - x1.iloc[i])**2 + (yP1 - y1.iloc[i])**2)
            B_P1 = m.sqrt((xP1 - xp2)**2 + (yP1 - yp2)**2)
            
            theta1 = 2*m.asin(P1_P2/(2*A_P1))
            theta2 = 2*m.asin(P1_P2/(2*B_P1))               
            
            As1 = 0.5*theta1*(r1.iloc[i])**2
            As2 = 0.5*theta2*(r2.iloc[i])**2
            
            h1 = m.sqrt(A_P1**2 - (0.5*P1_P2)**2)
            h2 = m.sqrt(B_P1**2 - (0.5*P1_P2)**2)
            
            At1 = 0.5*P1_P2*h1
            At2 = 0.5*P1_P2*h2
            
            A_overlap += As1 + As2 - At1 -At2

    return A_tot2/A_overlap

def f(params):
    
    df1 = pd.read_csv('4TSEPLFIM00471/b659f4d6-bbd2-4a74-8362-e948ec67fc81/mat_results/merged_fibres_all_um_4TSEPLFIM00471_SiPM_A.csv',sep="\t",header=None)
    df2 = pd.read_csv('ClearFiber1-L-P2/86fc5817-4dbb-4ddf-a38d-01c8bcac0206/mat_results/merged_fibres_all_um_ClearFiber1-L-P2_SiPM_A.csv',sep="\t",header=None)
    pins2 = pd.read_csv('ClearFiber1-L-P2/86fc5817-4dbb-4ddf-a38d-01c8bcac0206/mat_results/pins_all_um_ClearFiber1-L-P2_SiPM_A.csv', sep="\t",header=None)
    
    X = params[0]
    Y = params[1]
    theta = params[2]
    
    return overlap_area(df1, df2, pins2, X, Y, theta)

L = len(df2)
X_min, X_max = df2['x'].iloc[0], df2['x'].iloc[L-1]
Y_min, Y_max = df2['y'].iloc[L-1], df2['y'].iloc[0]
theta_min, theta_max = 0, 2*m.pi

initial_guess = [pins2['x'].iloc[0], 0, 0]
bounds = [(X_min, X_max), (Y_min, Y_max), (theta_min, theta_max)]

params_min = [X_min, Y_min, theta_min]
params_max = [X_max, Y_max, theta_max]

print(f(initial_guess))

#result = minimize(f, initial_guess, bounds=bounds)
#print(result)

