from re import L
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math as m
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
from scipy.optimize import basinhopping
from scipy.optimize import least_squares
from matplotlib.collections import PatchCollection
from shapely.geometry.point import Point
import geopandas as gpd
import descartes 
from shapely.ops import cascaded_union

from shapely.affinity import translate
from shapely.affinity import rotate

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

################################ FUNCTIONS ############################################################################################################################################################################
def draw_circles(circles1, circles2, x_min, x_max, y_min, y_max, title, name):
        fig, ax = plt.subplots()

        patches1 = descartes.PolygonPatch(circles1, fc='blue', ec="blue", alpha=0.5)
        patches2 = descartes.PolygonPatch(circles2, fc='red', ec="red", alpha=0.5)

        ax.add_patch(patches1)
        ax.add_patch(patches2)

        ax.set_xlim((x_min, x_max))
        ax.set_ylim((y_min, y_max))
        ax.set_xlabel('x [um]')
        ax.set_ylabel('y [um]')
        ax.set_title(title)
 
        fig.savefig('Plots/'+name+'.png')       
        fig.savefig('Plots/'+name+'.pdf')

def reflect(df):
    x_mean = df['x'].mean()
    df_reflected = df[['x']].copy()
    x_r = np.zeros(len(df))

    for i in range(len(df)):
        x = df['x'].iloc[i]
        y = df['y'].iloc[i]

        x = x -2*(x-x_mean)
        x_r[i] = x

    df_reflected['x'] = x_r
    df_reflected['y'] = df2['y']
    df_reflected['r'] = df2['r']
    return df_reflected

def get_intersections(circles1, circles2):
    return circles1.intersection(circles2)

def draw_intersections(intersections, x_min, x_max, y_min, y_max, title, name):
    
    fig, ax = plt.subplots()
    patches = descartes.PolygonPatch(intersections, fc='brown', alpha=.65)
    ax.add_patch(patches)

    ax.set_xlim((x_min, x_max))
    ax.set_ylim((y_min, y_max))
    ax.set_xlabel('x [um]')
    ax.set_ylabel('y [um]')
    ax.set_title(title)

    fig.savefig('Plots/'+name+'.png')       
    fig.savefig('Plots/'+name+'.pdf')

def get_overlap_area(circles1, circles2, intersections):
    A1 = circles1.area
    A2 = circles2.area
    A_overlap = intersections.area
    
    return (A_overlap/A1 , A_overlap/A2)

################################################################################################################################################################

print("Preparing the datasets")
df1 = pd.read_csv('Fiber_data/ClearFibreS1.csv',sep="\t",header=None)
df2 = pd.read_csv('Fiber_data/ClearFibreL1.csv',sep="\t",header=None)

df1.columns = ["index","x", "y", "x'", "y'","r","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","smth_last"]
df2.columns = ["index","x", "y", "x'", "y'","r","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","smth_last"]
df1.head()
df2.head()

print("Creating circles")
x1 = df1['x']
y1 = df1['y']
r1 = df1['r']
x2 = df2['x']
y2 = df2['y']
r2 = df2['r']

circle1 = []
for i in range(len(df1)):
    circle1.append( Point(x1.iloc[i], y1.iloc[i]).buffer(r1.iloc[i]) )
circles1 = cascaded_union(circle1)

circle2 = []
for i in range(len(df2)):
    circle2.append( Point(x2.iloc[i], y2.iloc[i]).buffer(r2.iloc[i]) )
circles2 = cascaded_union(circle2)

#draw_circles(circles1, circles2, -5000, 140000, -1500, 250, 'Initial configuration. 2nd mat not reflected', 'initial_config_S1_L1')

df2_reflected = reflect(df2)
x2_r = df2_reflected['x']
y2_r = df2_reflected['y']
r2_r = df2_reflected['r']

circle2_reflected = []
for i in range(len(df2_reflected)):
    circle2_reflected.append( Point(x2_r.iloc[i], y2_r.iloc[i]).buffer(r2_r.iloc[i]) )
circles2_reflected = cascaded_union(circle2_reflected)

#draw_circles(circles1, circles2_reflected, -5000, 140000, -1500, 250, 'Initial configuration. 2nd mat reflected', 'initial_config_S1_L1_reflected')
#draw_circles(circles1, circles2_reflected, -5000, 10000, -1500, 250, 'Initial configuration. 2nd mat reflected', 'initial_config_S1_L1_reflected_ZoomIn_left')
#draw_circles(circles1, circles2_reflected, 90000, 110000, -1500, 250, 'Initial configuration. 2nd mat reflected', 'initial_config_S1_L1_reflected_ZoomIn_right')

print("Finding intersections between the 2 mats")
intersections = get_intersections(circles1, circles2_reflected)

print("Drawing intersections")
#draw_intersections(intersections, -5000, 140000, -1500, 250, "Intersections. Initial config. 2nd mat reflected.", 'initial_config_intersections_S1_L1')
#draw_intersections(intersections, -5000, 10000, -1500, 250, "Intersections. Initial config. 2nd mat reflected.", 'initial_config_intersections_S1_L1_ZoomIn_left')
#draw_intersections(intersections, 90000, 110000, -1500, 250, "Intersections. Initial config. 2nd mat reflected.", 'initial_config_intersections_S1_L1_ZoomIn_right')

print("Getting area of intersections")
areas = get_overlap_area(circles1, circles2_reflected, intersections)
print("A_overlap/A1 = ", areas[0])
print("A_overlap/A2 = ", areas[1]) 

print("Maximising overlap area")
def function(params, circles1=circles1, circles2=circles2_reflected):

    X = params[0]
    Y = params[1]
    theta = params[2]

    # Transform 2nd mat (translation + rotation)
    translated = translate(circles2, xoff=X, yoff=Y)
    circles2_transformed = rotate(translated, theta, origin=Point(0.,0.), use_radians=True)

    intersections = get_intersections(circles1, circles2_transformed)

    areas = get_overlap_area(circles1, circles2, intersections)

    ratio1 = areas[0] # A_overlap/A1
    ratio2 = areas[1] # A_overlap/A2

    print('X = ', X, ' Y = ', Y, '  theta = ', theta, ' A_overlap/A1 = ', ratio1, ' A_overlap/A2 = ', ratio2)	
    if(ratio1 == 0):
        return 10
    else:
        return 1/ratio1

X_min, X_max = -2000, 2000
Y_min, Y_max = -500, 500
theta_min, theta_max = -m.pi, m.pi
initial_guess = [-(df2_reflected['x'].iloc[2304]-df1['x'].iloc[2]), 0., 0.]

result = least_squares(function, initial_guess, bounds=[(X_min, Y_min, theta_min), (X_max, Y_max, theta_max)])
print(result)

print("Drawing maximisation result")
new_params = result.x 
translated = translate(circles2_reflected, xoff=new_params[0], yoff=new_params[1])
circles2_result = rotate(translated, new_params[2], origin=Point(0.,0.), use_radians=True)
#draw_circles(circles1, circles2_result, -5000, 140000, -1500, 250, 'Final configuration.', 'final_config_S1_L1_reflected')
#draw_circles(circles1, circles2_result, -5000, 10000, -1500, 250, 'Final configuration.', 'final_config_S1_L1_reflected_ZoomIn_left')
#draw_circles(circles1, circles2_result, 90000, 110000, -1500, 250, 'Final configuration.', 'final_config_S1_L1_reflected_ZoomIn_right')

intersections_result = get_intersections(circles1, circles2_result)
draw_intersections(intersections_result, -5000, 140000, -1500, 250, "Intersections. Final config.", 'final_config_intersections_S1_L1')
draw_intersections(intersections_result, -5000, 10000, -1500, 250, "Intersections. Final config.", 'final_config_intersections_S1_L1_ZoomIn_left')
draw_intersections(intersections_result, 90000, 110000, -1500, 250, "Intersections. Final config.", 'final_config_intersections_S1_L1_ZoomIn_right')
