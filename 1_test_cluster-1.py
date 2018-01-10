from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pdb
import gmplot

GLOBAL_NUM_SAMPLES = 60
GLOBAL_NUM_FEATURES = 5
GLOBAL_USER_DATA_CSV_FILE = '00_test_user_data.csv'
GLOBAL_NUM_CLUSTERS = 3

def read_user_data_from_csv():
    user_data=np.zeros([GLOBAL_NUM_SAMPLES,GLOBAL_NUM_FEATURES])

    csv_file=open(GLOBAL_USER_DATA_CSV_FILE,'r')
    data = list(csv.DictReader(csv_file))

    i=0
    for item in data:
        if(item['dayofweek'] != 'Null'):
            user_data[i][0]=int(item['dayofweek'])
        if(item['startTime']!= 'Null'):
            user_data[i][1]=float(item['startTime'])
        if(item['stayingTime']!='Null'):
            user_data[i][2]=float(item['stayingTime'])
        if(item['longitude']!='Null'):
            user_data[i][3]=float(item['longitude'])
        if(item['latitude']!='Null'):
            user_data[i][4]=float(item['latitude'])
        i=i+1

    print('Orignal User Data:')
    print(user_data)
    return user_data

def extracted_features(user_data, feature_range):
    extracted_data = np.zeros([GLOBAL_NUM_SAMPLES,len(feature_range)])
    i = 0
    for item in feature_range:
        extracted_data[:,i] = user_data[:,item]
        i = i + 1
    return extracted_data

def seed_normalization(user_data):
    # normalization

    # minWeek = 1
    # maxWeek = 5
    # minStartTime = min(user_data[:,1])
    # maxStartTime = max(user_data[:,1])
    # minStayingTime = min(user_data[:,2])
    # maxStayingime = max(user_data[:,2])
    # minLon = min(user_data[:,3])
    # maxLon = max(user_data[:,3])
    # minLat = min(user_data[:,4])
    # maxLat = max(user_data[:,4])
    # def list_norm(valueList, minValue, maxValue):
    #     i = 0
    #     for x in np.nditer(valueList):
    #         valueList[i] = (x - minValue)/(maxValue-minValue)
    #         i = i + 1

    # list_norm(user_data[:,0], minWeek, maxWeek)
    # list_norm(user_data[:,1], minStartTime, maxStartTime)
    # list_norm(user_data[:,2], minStayingTime, maxStayingime)
    # list_norm(user_data[:,3], minLon, maxLon)
    # list_norm(user_data[:,4], minLat, maxLat)  
    user_data_norm = (user_data - user_data.min()) / (user_data.max() - user_data.min()) 

    print('Below printing user_data after normalization:')
    print(user_data)
    print(user_data_norm)
    return user_data_norm

def cluster_by_seed_timeAndLocation():   
    user_data = read_user_data_from_csv()
    user_data_norm = seed_normalization(user_data)

    # user_data: dayofweek, startTime,stayingTime,longitude,latitude
    # extracted_data: startTime,stayingTime,longitude,latitude
    consider_features = [1, 2, 3, 4]
    pdb.set_trace()
    extracted_data = extracted_features(user_data_norm, consider_features)

    print('extracted_data for KMeans: startTime,stayingTime,longitude,latitude ')
    for item in extracted_data:
        print '%0.3f' % item[0], '%0.3f' % item[1], '%0.5f' % item[2], '%0.5f' % item[3]

    kmeans = KMeans(n_clusters = 3)
    #kmeans = KMeans()
    kmeans.fit(extracted_data)

    output3D = np.zeros([GLOBAL_NUM_SAMPLES,len(consider_features)])
    output2D = np.zeros([GLOBAL_NUM_SAMPLES])
    output3D[:,0] = extracted_data[:,0]
    output3D[:,1] = extracted_data[:,1]
    output3D[:,2] = extracted_data[:,2]
    output3D[:,3] = kmeans.predict(extracted_data)
    output2D = kmeans.predict(extracted_data)
    print(output3D)

    df = pd.DataFrame(output3D, columns=['Feature1', 'Feature2','Feature3','Cluster'])

    #################################### plot in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.array(df['Feature1'])
    y = np.array(df['Feature2'])
    z = np.array(df['Feature3'])
    damn=['start time','staying time', 'longitude']
    ax.scatter(x,y,z, marker="o", c=df['Cluster'], s=50, label=damn,cmap="RdBu")
    ax.set_xlabel('start time')
    ax.set_ylabel('staying time')
    ax.set_zlabel('longitude')

    # labels
    labels = kmeans.labels_
    lable_unique = np.unique(labels)
    num_cluster = len(lable_unique)
    print(labels)
    print(lable_unique)
    print(num_cluster)

    #centers
    centers = kmeans.cluster_centers_
    print(centers)
    ax.scatter(centers[:, 0], centers[:, 1], centers[:,2], marker = '*', c='r', s=200, alpha=1)
    #end
    plt.title('Cluster of staying places')
    plt.show()

def cluster_by_seed_location():
    user_data = read_user_data_from_csv()
    user_data_norm = seed_normalization(user_data)

    consider_features = [3, 4]
    extracted_data = extracted_features(user_data_norm, consider_features)

    #kmeans = KMeans(n_clusters = 3)
    kmeans = KMeans()
    kmeans.fit(extracted_data)
    output2D = np.zeros([GLOBAL_NUM_SAMPLES])
    output2D = kmeans.fit_predict(extracted_data)
    print(output2D)

    pdb.set_trace()
    plt.figure(1)
    plt.scatter(extracted_data[:,0], extracted_data[:,1], c = output2D)
    plt.show()

    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], marker = '*', c = 'b', s = 200, alpha = 0.5)

    plt.title('Cluster of locations(longitude, latitude)')
    plt.show()

def draw_locations_on_GoogleMap():
    user_data = read_user_data_from_csv()
    coords_features = [3,4]
    coords = extracted_features(user_data, coords_features)

    lats = []
    longs = []
    for item in coords:
        lats.append(float(item[0]))
        longs.append(float(item[1]))
    # Reference point
    gmap = gmplot.GoogleMapPlotter(lats[0], longs[0], 16)
    # Draw all points
    gmap.scatter(lats, longs, 'red', size = 50, marker = True)

    # Draw circle for each cluster
    consider_features = [3, 4]
    extracted_data = extracted_features(user_data, consider_features)
    #kmeans = KMeans(n_clusters = 3)
    kmeans = KMeans()
    kmeans.fit(extracted_data)
    output2D = np.zeros([GLOBAL_NUM_SAMPLES])
    output2D = kmeans.fit_predict(extracted_data)
    centers_ = kmeans.cluster_centers_
    #cluster_center = [float(item[0]), float(item[1]) for item in centers_]
    for item in centers_:
        gmap.circle(float(item[0]), float(item[1]), radius = 1000, color = 'yellow')

    #gmap.heatmap(lats, longs)
    gmap.draw("test_map.html")

def main():
    #main function
    # version 1: cluster the user data by seed of (start time, staying time, location(longitude, latitude))
    #cluster_by_seed_timeAndLocation()
    
    # version 2: cluster the user data only by seed of location (longitude, latitude)
    # 1. cluster by location
    # 2. draw location clusters on map
    # todo 3. assign time zone to different location
    # todo 4. show the user pattern by weekly time zones
    #cluster_by_seed_location()
    draw_locations_on_GoogleMap()

main()
