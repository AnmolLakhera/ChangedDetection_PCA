import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import Counter

def find_vector_set(diff_image, new_size):
    vector_set = []
    for i in range(0, new_size[0], 5):
        for j in range(0, new_size[1], 5):
            block = diff_image[i:i+5, j:j+5]
            feature = block.flatten()
            vector_set.append(feature)
    vector_set = np.array(vector_set)
    
    mean_vec = np.mean(vector_set, axis=0)    
    vector_set = vector_set - mean_vec
    
    return vector_set, mean_vec

def find_FVS(EVS, diff_image, mean_vec, new_size):
    feature_vector_set = []
    for i in range(2, new_size[0]-2):
        for j in range(2, new_size[1]-2):
            block = diff_image[i-2:i+3, j-2:j+3]
            feature = block.flatten()
            feature_vector_set.append(feature)
    feature_vector_set = np.array(feature_vector_set)
    
    FVS = np.dot(feature_vector_set, EVS.T)
    FVS = FVS - mean_vec
    return FVS

def clustering(FVS, components, new_size):
    kmeans = KMeans(n_clusters=components, verbose=0)
    kmeans.fit(FVS)
    output = kmeans.predict(FVS)
    count = Counter(output)

    least_index = min(count, key=count.get)
    change_map = np.reshape(output, (new_size[0]-4, new_size[1]-4))
    
    return least_index, change_map

def find_PCAKmeans(imagepath1, imagepath2):
    image1 = cv2.imread(imagepath1)
    image2 = cv2.imread(imagepath2)
    new_size = np.asarray(image1.shape[:2]) // 5 * 5
    image1 = cv2.resize(image1, (new_size[1], new_size[0]))
    image2 = cv2.resize(image2, (new_size[1], new_size[0]))
    
    diff_image = cv2.absdiff(image1, image2)
    cv2.imwrite('diff3.jpg', diff_image)
    
    vector_set, mean_vec = find_vector_set(diff_image, new_size)
    
    pca = PCA()
    pca.fit(vector_set)
    EVS = pca.components_
    
    FVS = find_FVS(EVS, diff_image, mean_vec, new_size)
    
    components = 3
    least_index, change_map = clustering(FVS, components, new_size)
    
    change_map[change_map == least_index] = 255
    change_map[change_map != 255] = 0
    change_map = change_map.astype(np.uint8)
    
    kernel = np.array(((0,0,1,0,0),
                       (0,1,1,1,0),
                       (1,1,1,1,1),
                       (0,1,1,1,0),
                       (0,0,1,0,0)), dtype=np.uint8)
    
    cleanChangeMap = cv2.erode(change_map, kernel)
    cv2.imwrite("changemap3.jpg", change_map)
    cv2.imwrite("cleanchangemap3.jpg", cleanChangeMap)

if __name__ == "__main__":
    a=r"C:\Users\missv\Desktop\New folder\Change-Detection-in-Satellite-Imagery\31.jpg"
    b=r"C:\Users\missv\Desktop\New folder\Change-Detection-in-Satellite-Imagery\32.jpg"
    find_PCAKmeans(a,b)