import math
import googlemaps
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from geopy.distance import vincenty
from math import acos
from math import sqrt
from math import pi
import utm
from math import atan2,degrees
import pandas

def GetAngleOfLineBetweenTwoPoints(p1, p2):
    xDiff = p2[0] - p1[0]
    yDiff = p2[1] - p1[1]
    return degrees(atan2(yDiff, xDiff)) 
    
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

topo_data = []
brngs = 12
brng_degree = 30
distance_iterations = 10
distance = 25

def toRad(numba):
    return numba * math.pi / 180
def toDeg(numba):
    return numba * 180 / math.pi

def destPts(lat, lng, brng, dist):
    dist = float(dist)/6371000 #earths radius in meters
    brng = toRad(brng)

    lat1 = toRad(lat)
    lon1 = toRad(lng)

    lat2 = math.asin(math.sin(lat1) * math.cos(dist) + math.cos(lat1) * math.sin(dist) * math.cos(brng))
    lon2 = lon1 + math.atan2(math.sin(brng) * math.sin(dist) * math.cos(lat1), math.cos(dist) - math.sin(lat1) * math.sin(lat2))
    # print {'lat2':toDeg(lat2), 'lng2':lon2}
    return {'lat2':toDeg(lat2), 'lng2':toDeg(lon2)}

    if (math.isnan(lat2) or math.isnan(lon2)):
        return null

def buildPts(lat, lng):
    for i in range(brngs): #brngs
        for j in range(1, distance_iterations): #iterations of distance
            newPts = destPts(lat, lng, i*brng_degree, j*distance) #30 degree brng, 25 meters distance
            topo_data.append((newPts.get('lat2'), newPts.get('lng2')))
            
gmaps_geo = googlemaps.Client(key='AIzaSyDWx2I0N0O9Hs97eELeROWqJvf0Uqj7jJM')
gmaps_elv = googlemaps.Client(key='AIzaSyATNfh9_TS3ABNySxZU0vTWknyULPCDEXM')

ui = input("Please enter an address to look up (1 for demo): ");

if ui == "1":
    print("2425 Ellentown Rd, La Jolla, CA")
    geocode_result = gmaps_geo.geocode("2425 Ellentown Rd, La Jolla, CA")
else:
    geocode_result = gmaps_geo.geocode(ui)


lat = geocode_result[0]['geometry']['location']['lat']
lng = geocode_result[0]['geometry']['location']['lng']

buildPts(lat, lng)

df = pandas.read_csv('CoreLogicData.csv', low_memory=False)

elim_radius = 0.0001
num_entries = 0
newdf = df[(lat-elim_radius < df['PARCEL LEVEL LATITUDE']) & (df['PARCEL LEVEL LATITUDE'] < lat+elim_radius) & (lng-elim_radius < df['PARCEL LEVEL LONGITUDE']) & (df['PARCEL LEVEL LONGITUDE']< lng+elim_radius)]
num_entries = len(newdf)
while num_entries < 100:
    newdf = df[(lat-elim_radius < df['PARCEL LEVEL LATITUDE']) & (df['PARCEL LEVEL LATITUDE'] < lat+elim_radius) & (lng-elim_radius < df['PARCEL LEVEL LONGITUDE']) & (df['PARCEL LEVEL LONGITUDE']< lng+elim_radius)]
    num_entries = len(newdf)
    elim_radius += 0.0001


Format = ['PARCEL LEVEL LATITUDE', 'PARCEL LEVEL LONGITUDE', 'FRONT FOOTAGE','LAND SQUARE FOOTAGE','UNIVERSAL-BUILDING-SQUARE-FEET']

df_selected = newdf[Format]

points = [(lat, lng)]
deets = []

for i in range(len(df_selected.index)):
    if i > 0:
        points.append((df_selected.iloc[i][0], df_selected.iloc[i][1]))
        deets.append([df_selected.iloc[i][2], df_selected.iloc[i][3], df_selected.iloc[i][4]])


elevations = gmaps_elv.elevation(points)

topos = gmaps_elv.elevation(topo_data)

print(topos)

lngs = []
lats = []
elvs = []

topo_lat = []
topo_lng = []
topo_elv = []

for j in range(len(elevations)):
    lats.append(elevations[j]['location']['lat'])
    lngs.append(elevations[j]['location']['lng'])
    elvs.append(elevations[j]['elevation'])
    
for k in range(len(topos)):
    topo_lat.append(topos[k]['location']['lat'])
    topo_lng.append(topos[k]['location']['lng'])
    topo_elv.append(topos[k]['elevation'])
    
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x_points = [[],[],[]]
y_points = [[],[],[]]
el_points = [[],[],[]]
selectedDeets = []

all_x = []
all_y = []

blockout_radius = 1

#print((lat, lng))

utm_coords = []
utm_topos = []

utm_og_house = utm.from_latlon(lat, lng)
#utm_og_house = (lat, lng)

for i in range(len(lngs)):
    utm_coords.append(utm.from_latlon(lats[i], lngs[i]))
    
for i in range(len(topo_lng)):
    utm_topos.append(utm.from_latlon(topo_lat[i], topo_lng[i]))
    
lowestEl = 0
lowestIdx = 0
highestEl = 0
highestIdx = 0
for i in range(len(lngs)):
    if i == 0:
        lowestEl = elvs[i]
        highestEl = elvs[i]
    else:
        if elvs[i] < lowestEl:
            lowestEl = elvs[i]
            lowestIdx = i
        elif elvs[i] > highestEl:
            highestEl = elvs[i]
            highestIdx = i

pt1 = (utm_og_house[0], utm_og_house[1])
pt2 = (utm_coords[lowestIdx][0], utm_coords[lowestIdx][1])
pt3 = (utm_coords[highestIdx][0], utm_coords[highestIdx][1]) 
offset1 = GetAngleOfLineBetweenTwoPoints(pt1, pt2)
offset2 = GetAngleOfLineBetweenTwoPoints(pt1, pt3)


offset2 = 180 + offset2

if offset1 < 0:
    offset1 = 360 + offset1

offset = (offset1 + offset2)/2

for i in range(len(lngs)):
    all_x.append(utm_coords[i][0])
    all_y.append(utm_coords[i][1])
    if utm_coords[i][0] <= utm_og_house[0] + blockout_radius and utm_coords[i][1] <= utm_og_house[1] + blockout_radius and utm_coords[i][0] >= utm_og_house[0] - blockout_radius and utm_coords[i][1] >= utm_og_house[1] - blockout_radius:
        x_points[2].append(utm_coords[i][0])
        y_points[2].append(utm_coords[i][1])
        el_points[2].append(elvs[i])
        pt1 = (utm_og_house[0], utm_og_house[1])
        pt2 = (utm_coords[i][0], utm_coords[i][1])
        #print("hello")
        #print(angle_clockwise(pt1, pt2))
        #print(pt1)
    else:
        pt1 = (utm_og_house[0], utm_og_house[1])
        pt2 = (utm_coords[i][0], utm_coords[i][1]) 
        
        angle = GetAngleOfLineBetweenTwoPoints(pt1, pt2)
        
        #angle < 75.0 and angle > -75:            
            
        #offset = 170
        view_angle = 60
        
        if angle < 0:
            angle = 360 + angle
            
        #print(angle)
            
        if angle < view_angle + offset and angle > -view_angle + offset:
            x_points[0].append(utm_coords[i][0])
            y_points[0].append(utm_coords[i][1])
            el_points[0].append(elvs[i])
            selectedDeets.append(deets[i])
        else:
            x_points[1].append(utm_coords[i][0])
            y_points[1].append(utm_coords[i][1])
            el_points[1].append(elvs[i])
        
        
colors = ['r','b','g','g']

#syn0 = []
#syn1 = []##
#
#f0 = open("syn0.txt", "r")
#for line in f0:
#    tempArry = line.strip().split(',')
#    #print(tempArry)
#    x = np.array(tempArry, dtype='|S4')
#    y = x.astype(np.float)
#    syn0.append(y)
#f1 = open("syn1.txt", "r")
#for line in f1:
#    tempArry = line.strip().split(',')
#    #print(tempArry)
#    x = np.array(tempArry, dtype='|S4')
#    y = x.astype(np.float)
#    syn1.append(y)

#print(syn0)
#print(syn1)

#l0 = selectedDeets
#l1 = nonlin(np.dot(l0,syn0))
#l2 = nonlin(np.dot(l1,syn1))

#print(l2)

x_topo = []
y_topo = []

for i in range(len(utm_topos)):
    x_topo.append(utm_topos[i][0])
    y_topo.append(utm_topos[i][1])    

for k in range(3):
    ax.scatter(x_points[k], y_points[k], el_points[k], c=colors[k], marker='o')    

#print(len(x_topo))
#print(len(y_topo))
#print(len(topo_elv))
ax.scatter(x_topo, y_topo, topo_elv, c='y', marker='*')
    
ax.set_xlabel('Latitude')
ax.set_ylabel('Longitude')
ax.set_zlabel('Elevation')

#plt.plot(np.unique(all_x), np.poly1d(np.polyfit(all_x, all_y, 1))(np.unique(all_x)))

plt.show()



