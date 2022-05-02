import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math as m
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
from scipy.optimize import basinhopping
from scipy.optimize import least_squares
from matplotlib.collections import PatchCollection

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def draw_circles(df1, df2, x_min, x_max, y_min, y_max, title, name):
        # 2D plot of the DataFrames    

        x1 = df1['x']
        y1 = df1['y']
        r1 = df1['r']
        x2 = df2['x']
        y2 = df2['y']
        r2 = df2['r']

        if(len(df1) < len(df2)): L = len(df1)
        else: L = len(df2)

        patches1 = [plt.Circle( (x1.iloc[i], y1.iloc[i]), r1.iloc[i], color='blue', alpha=0.5) for i in range(L)]
        patches2 = [plt.Circle( (x2.iloc[i], y2.iloc[i]), r1.iloc[i], color='red', alpha=0.5) for i in range(L)]

        fig, ax = plt.subplots()

        ax = plt.gca()
        ax.cla()
        ax.set_xlim((x_min, x_max))
        ax.set_ylim((y_min, y_max))
        ax.set_xlabel('x [um]')
        ax.set_ylabel('y [um]')
        ax.set_title(title)

        col1 = PatchCollection(patches1, facecolors='blue')
        col2 = PatchCollection(patches2, facecolors='red')

        ax.add_collection(col1)
        ax.add_collection(col2)
 
        fig.savefig('Plots/'+name+'.png')       
        fig.savefig('Plots/'+name+'.pdf')

### 1) Preparing the DataFrames
df1 = pd.read_csv('4TSEPLFIM00471/b659f4d6-bbd2-4a74-8362-e948ec67fc81/mat_results/merged_fibres_all_um_4TSEPLFIM00471_SiPM_A.csv',sep="\t",header=None) # Sci-Fi 
df2 = pd.read_csv('ClearFiber1-L-P2/86fc5817-4dbb-4ddf-a38d-01c8bcac0206/mat_results/merged_fibres_all_um_ClearFiber1-L-P2_SiPM_A.csv',sep="\t",header=None) # clear fibers
pins2 = pd.read_csv('ClearFiber1-L-P2/86fc5817-4dbb-4ddf-a38d-01c8bcac0206/mat_results/pins_all_um_ClearFiber1-L-P2_SiPM_A.csv', sep="\t",header=None) # pins positions (clear fibres)

df1.columns = ["index","x", "y", "x'", "y'","r","smth0","smth1","smth2","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","smth_last"]
df2.columns = ["index","x", "y", "x'", "y'","r","smth0","smth1","smth2","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","smth_last"]
df1.head()
df2.head()

pins2.columns = ["index", "x", "y", "x'","y'","r","smth0","smth_last"]
pins2.head()

df2.drop(df2.index[2850:2862], axis=0, inplace=True) # so that the two mats have the same number of fibers

def reflect(df):
	x_r = np.zeros(len(df))
	df_reflected = df[['x']].copy()

	for i in range(0, len(df), 6):
		df_subset = df.loc[len(df)-6-i:len(df)-1-i].copy().reset_index(drop=True)

		x_r[i] = df_subset['x'].iloc[0]
		x_r[i+1] = df_subset['x'].iloc[1]
		x_r[i+2] = df_subset['x'].iloc[2]
		x_r[i+3] = df_subset['x'].iloc[3]
		x_r[i+4] = df_subset['x'].iloc[4]
		x_r[i+5] = df_subset['x'].iloc[5]
		
	df_reflected['x'] = x_r
	df_reflected['y'] = df2['y']
	df_reflected['r'] = df2['r']
	return df_reflected

df2_reflected = reflect(df2)
draw_circles(df2, df2_reflected, -70000, 70000, -5600, -4000, 'Before (blue) and after (red) reflection', 'before_after_reflection')

# The way fibers are numbered must be consistent in the two mats. Numbering: starts form lowest x-value (from top to bottom layer) and goes on to higher x values (from top to bottom layer).
# The raw dataframes start out as numbered from lowest to highest x-value. But this can be incosistent among the two mats.
def sort(df):
        df_sorted = pd.DataFrame(columns=['x','y','r'])
        for i in range(0,len(df)-5,6):
                df_subset = df.loc[i:i+5].copy()
                df_subset = df_subset.sort_values('y',ascending=False) # sorts in y in increments of 6 (there are 6 layers)
                df_subset = df_subset.reset_index(drop=True)

                df_sorted = df_sorted.append({'x' : df_subset['x'].iloc[i-(i-0)], 'y' : df_subset['y'].iloc[i-(i-0)], 'r' : df_subset['r'].iloc[i-(i-0)]}, ignore_index = True)
                df_sorted = df_sorted.append({'x' : df_subset['x'].iloc[i-(i-1)], 'y' : df_subset['y'].iloc[i-(i-1)], 'r' : df_subset['r'].iloc[i-(i-1)]}, ignore_index = True)
                df_sorted = df_sorted.append({'x' : df_subset['x'].iloc[i-(i-2)], 'y' : df_subset['y'].iloc[i-(i-2)], 'r' : df_subset['r'].iloc[i-(i-2)]}, ignore_index = True)
                df_sorted = df_sorted.append({'x' : df_subset['x'].iloc[i-(i-3)], 'y' : df_subset['y'].iloc[i-(i-3)], 'r' : df_subset['r'].iloc[i-(i-3)]}, ignore_index = True)
                df_sorted = df_sorted.append({'x' : df_subset['x'].iloc[i-(i-4)], 'y' : df_subset['y'].iloc[i-(i-4)], 'r' : df_subset['r'].iloc[i-(i-4)]}, ignore_index = True)
                df_sorted = df_sorted.append({'x' : df_subset['x'].iloc[i-(i-5)], 'y' : df_subset['y'].iloc[i-(i-5)], 'r' : df_subset['r'].iloc[i-(i-5)]}, ignore_index = True)

        return df_sorted

df1 = sort(df1)
df2_reflected = sort(df2_reflected)
### DataFrame preparation ends 

### 2) Computing the overlap area between the 2 mats
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

def rotate(x, y, theta, D):

    # translate origin to (X,Y)
    x1 = x + D/2
    
    # rotate by theta
    x2 = m.cos(theta)*x1 + m.sin(theta)*y
    y2 = -m.sin(theta)*x1 + m.cos(theta)*y

    # translate orgin back to its position
    x2_t = x2 - D/2

    return (x2_t, y2)
    
def transform(x, y, mark_x1, D, X, Y, theta):
    """ Makes transformation to new position of the alignements pins. x1 is the x-coordinate of the 1st mark. """
    return rotate( translate(x, y, mark_x1, X, Y)[0], translate(x, y, mark_x1, X, Y)[1], theta, D)
    
def overlap_area(df1, df2, pins2, X, Y, theta):
	""" Calculates the overlap area between fiber mat 1 (df1) and reflected fiber mat 2 (df2) """
	x1 = df1['x']
	x2 = df2['x']
	y1 = df1['y']
	y2 = df2['y']
	r1 = df1['r']
	r2 = df2['r']
	mark_x1 = pins2['x'].iloc[0]

	if(len(df1) < len(df2)): L = len(df1)
	else: L = len(df2)
	D = pins2['x'].iloc[2] - pins2['x'].iloc[0]

	# Compute overlap area
	A_overlap = 0
	A_tot1 = 0
	A_tot2 = 0
	
	n = 0
	for i in range(L):
		if( (i % 6 == 0) and (i != 0) ): n+= 6
		
		A1 = m.pi*(r1.iloc[i])**2
		A2 = m.pi*(r2.iloc[i])**2

		A_tot1 += A1
		A_tot2 += A2

		r = 30
		if(i - r < 0): 
			j_min = 0
			j_max = i+r
		elif(i + r > L): 
			j_min = i-r
			j_max = L-r
		else:
			j_min = i-r
			j_max = i+r

		#print("(x1, y1) = ", x1.iloc[i], y1.iloc[i])

		#for j in range(j_min,j_max,6):

			#print("(i,j) = ", i, L-(n+6-(j-n)) )
			#print("(x2, y2) = ", x2.iloc[L-6-j], y2.iloc[L-6-j])


		# Transform 2nd mat to a new reference frame, where we will put the alignment pins
		(xp2, yp2) = transform(x2.iloc[L-(n+6-(i-n))], y2.iloc[L-(n+6-(i-n))], mark_x1, D, X, Y, theta) 	
		#print("(x2, y2) = ", x2.iloc[L-(n+6-(i-n))], y2.iloc[L-(n+6-(i-n))])
		#print("(x2p, y2p) = ", xp2, yp2)

		# Find the intersections between the 2 mats
		intersections = find_intersections(x1.iloc[i], y1.iloc[i], xp2, yp2, r1.iloc[i], r2.iloc[L-(n+6-(i-n))])		

		if intersections == None:
			continue
		elif intersections == False:
			if(A1 < A2): A_overlap += A1
			else: A_overlap += A2
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

			if(abs(P1_P2/(2*A_P1))>1): continue
			if(abs(P1_P2/(2*B_P1))>1): continue          			 
			theta1 = 2*m.asin(P1_P2/(2*A_P1))
			theta2 = 2*m.asin(P1_P2/(2*B_P1))               
         	  
			As1 = 0.5*theta1*(r1.iloc[i])**2
			As2 = 0.5*theta2*(r2.iloc[i])**2
          
			h1 = m.sqrt(A_P1**2 - (0.5*P1_P2)**2)
			h2 = m.sqrt(B_P1**2 - (0.5*P1_P2)**2)
          
			At1 = 0.5*P1_P2*h1
			At2 = 0.5*P1_P2*h2
           	
			A_overlap += As1 + As2 - At1 -At2

	if(A_overlap == 0): return 10
	else: return A_tot2/A_overlap

draw_circles(df1, df2_reflected, -70000, 70000, -5600, -4000, 'Initial configuration', 'initial_configuration')
x0 = [pins2['x'].iloc[0], pins2['y'].iloc[0], 0.]
#A0 = overlap_area(df1, df2_reflected, pins2, x0[0], x0[1], x0[2])
#print(A0)
### Computation of overlap area ends

### 3) Minimisation
def f(params, df1 = df1, df2_reflected = df2_reflected, pins2 = pins2):    

	X = params[0]
	Y = params[1]
	theta = params[2]

	print('X = ', params[0], ' Y = ', params[1], '  theta = ', params[2], ' f(params) = ', overlap_area(df1, df2_reflected, pins2, X, Y, theta))	
	return overlap_area(df1, df2_reflected, pins2, X, Y, theta)

# Boundaries
X_min, X_max = pins2['x'].iloc[0]-500, pins2['x'].iloc[0]+500
Y_min, Y_max = -500, 100
theta_min, theta_max = -m.pi, m.pi
bounds = [(X_min, X_max), (Y_min, Y_max), (theta_min, theta_max)]

# Initial guess
initial_guess = [pins2['x'].iloc[0] - abs(df2_reflected['x'].iloc[len(df2_reflected)-6]-df1['x'].iloc[0]), pins2['y'].iloc[0] - abs(df2_reflected['y'].iloc[len(df2_reflected)-6]-df1['y'].iloc[0]), 0.]
#f(initial_guess)

df2_transformed = df2_reflected[['x']].copy()
mark_x1 = pins2['x'].iloc[0]
D = pins2['x'].iloc[2] - pins2['x'].iloc[0]

x_trans = np.zeros(len(df2_reflected))
y_trans = np.zeros(len(df2_reflected))

for i in range(len(df2_reflected)):
        (x_trans[i], y_trans[i]) = transform(df2_reflected['x'].iloc[i], df2_reflected['y'].iloc[i], mark_x1, D, initial_guess[0], initial_guess[1], initial_guess[2])

df2_transformed['x'] = x_trans
df2_transformed['y'] = y_trans
df2_transformed['r'] = df2_reflected['r'].copy()

draw_circles(df1, df2_transformed, -70000, 70000, -5600, -4000, 'Initial guess', 'initial_guess') 

params_min = [X_min, Y_min, theta_min]
params_max = [Y_max, Y_max, theta_max]

# Minimization

#result = minimize(f,initial_guess,bounds=bounds)
result = least_squares(f, initial_guess, bounds=[(X_min, Y_min, theta_min), (X_max, Y_max, theta_max)])
#result = basinhopping(f, initial_guess, T=0.1)
print(result)

new_params = result.x 
df4 = df2_reflected[['x']].copy()
mark_x1 = pins2['x'].iloc[0]
D = pins2['x'].iloc[2] - pins2['x'].iloc[0]

x_trans = np.zeros(len(df2_reflected))
y_trans = np.zeros(len(df2_reflected))

for i in range(len(df2)):
	(x_trans[i], y_trans[i]) = transform(df2_reflected['x'].iloc[i], df2_reflected['y'].iloc[i], mark_x1, D, new_params[0], new_params[1], new_params[2])

df4['x'] = x_trans
df4['y'] = y_trans
df4['r'] = df2_reflected['r'].copy()

draw_circles(df1, df4, -70000, 70000, -5600, -4000, 'Final configuration', 'final_configuration')