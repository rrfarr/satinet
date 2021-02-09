import os
import argparse
import imageio
import subprocess
import rasterio as rio
import numpy as np
import warnings
from libs2p import s2p as s2p
import bs4
import pyproj
import math
import shutil
import pickle
import libs2p.utils as utils
import libs2p.rectification as rec
from tqdm import tqdm
from skimage import io as io #numpy version has to be ==1.15.0
import libmccnn
import matplotlib.pyplot as plt
import tensorflow as tf

from libs2p import utils as utils
from libs2p import pointing_accuracy as pointing_accuracy

registrationDelta = 5
zGraphSpan = -1
INVALID_Z = -9999
COMPLETENESS_THRESHOLD = 1
DEFAULT_SPACING = 0.3

'''   
def mccnn_basic_stereo_vision(img_left_filename, img_right_filename,disp_min,disp_max,tag,mccnn_model_path,sub_pixel_interpolation):
    
    patch_size=11
    patch_height = patch_size
    patch_width = patch_size
    resume =  os.path.join(mccnn_model_path, 'checkpoint')

    ####################
    # do matching
    # get data path
    left_path = img_left_filename
    right_path = img_right_filename
    
    # reading images
    left_image = io.imread(left_path).astype(np.float32)
    right_image = io.imread(right_path).astype(np.float32)
    left_image = (left_image - np.mean(left_image, axis=(0, 1))) / np.std(left_image, axis=(0, 1))
    right_image = (right_image - np.mean(right_image, axis=(0, 1))) / np.std(right_image, axis=(0, 1))
    left_image = np.expand_dims(left_image, axis=2)
    right_image = np.expand_dims(right_image, axis=2)
    
    # Derive the dimensions of the two images
    height, width = left_image.shape[:2]

    # Compute the left and right features
    featuresl, featuresr = libmccnn.process_functional_Bi.compute_features(left_image, right_image, None, None, patch_height, patch_width, resume,1)
    
    print('Construct the left and right cost volumes ...')
    left_cost_volume, right_cost_volume = libmccnn.process_functional_Bi.compute_cost_volume(featuresl,featuresr,disp_min, disp_max)
    
    # Apply a median filter to the cost volumes
    left_cost_volume = median_filter(left_cost_volume)
    right_cost_volume = median_filter(right_cost_volume)

    # Compute the left disparity map
    print('Compute the left disparity map')
    left_disparity_map = libmccnn.process_functional_Bi.disparity_selection(left_cost_volume, np.arange(disp_min, disp_max+1,1),sub_pixel_interpolation)
    
    # Compute the right disparity map
    print('Compute the right disparity map')
    right_disparity_map = -libmccnn.process_functional_Bi.disparity_selection(right_cost_volume, np.arange(disp_min, disp_max+1,1),sub_pixel_interpolation)
    
    # Estimate the disparity map aligned with the left view
    print('Compute the left right consistency')
    disp = libmccnn.process_functional_Bi.left_right_consistency(left_disparity_map, right_disparity_map)

    out_disp_foldername = os.path.join('./data/',tag)

    # load the model parameters
    model = np.load(os.path.join(out_disp_foldername,'model.npy'))
    
    # Extract the slope and intercept parameter
    slope = model[0]
    intercept = model[1]

    # Derive the height map
    height_map = np.full((disp.shape),np.nan)
    
    #height[np.isnan(disp_in)] = np.nan
    height_map[~np.isnan(disp)] = disp[~np.isnan(disp)] * slope + intercept
    
    return height_map
'''
from skimage.transform import warp, AffineTransform

def warp_feature(feature, tform):
    # Determine the number of feature maps
    Nfeatures = feature.shape[2]
    # Initialize the feature translated
    feature_d = np.zeros(feature.shape)
    
    # Translate the nth feature - one at a time
    for n in range(Nfeatures):
        feature_d[:,:,n] = warp(feature[:,:,n], tform)
    # Return the translated features
    return feature_d
    

def  sgbm_stereo_vision(out_foldername,img_left_filename, img_right_filename,disp_min,disp_max,tag):
    warnings.filterwarnings("ignore")
    # opencv sgbm function implements a modified version of Hirschmuller's
    # Semi-Global Matching (SGM) algorithm described in "Stereo Processing
    # by Semiglobal Matching and Mutual Information", PAMI, 2008
       
    # Specify where the rectified disparity map will be stored
    disp_filename = os.path.join(out_foldername, 'rectified_disp.tif')

    p1 = 8  # penalizes disparity changes of 1 between neighbor pixels
    p2 = 32  # penalizes disparity changes of more than 1
    # it is required that p2 > p1. The larger p1, p2, the smoother the disparity

    win = 3  # matched block size. It must be a positive odd number
    lr = 1  # maximum difference allowed in the left-right disparity check
        
    # Create a temporary directory
    cost_filename = os.path.join(out_foldername,'temp.tif')
        
    # Derive the command line to be used to execute sgbm
    cmd = 'sgbm {} {} {} {} {} {} {} {} {} {}'.format(img_left_filename, img_right_filename,
                                                           disp_filename, cost_filename,
                                                           disp_min,
                                                           disp_max,
                                                           win, p1, p2, lr)
    # Execute the system command to compute SGBM
    print('Computing the SGBM stereo vision method ...')
    subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT) 
        
    print('Cleaning and saving disparity as .npy file ...')

    # Remove the temp.tif file
    os.remove(cost_filename)
        
    # Load the disparity data
    with rio.open(disp_filename) as img :
        disp= img.read().squeeze()
    # Set the polarity to negative values
    disp = -disp

    # Remove the disparity map stored in tif file
    os.remove(disp_filename)

    out_disp_foldername = os.path.join('./data/',tag)

    # load the model parameters
    model = np.load(os.path.join(out_disp_foldername,'model.npy'))
    
    # Extract the slope and intercept parameter
    slope = model[0]
    intercept = model[1]

    # Derive the height map
    height_map = np.full((disp.shape),np.nan)
    
    #height[np.isnan(disp_in)] = np.nan
    height_map[~np.isnan(disp)] = disp[~np.isnan(disp)] * slope + intercept
    
    return height_map
    

def compute_utm_zone(lon, lat):
    """
    Compute the UTM zone which contains
    the point with given longitude and latitude

    Args:
        lon (float): longitude of the point
        lat (float): latitude of the point

    Returns:
        str: UTM zone number + hemisphere (eg: '30N')
    """
    # UTM zone number starts from 1 at longitude -180,
    # and increments by 1 every 6 degrees of longitude
    zone = int((lon + 180) // 6 + 1)

    hemisphere = "N" if lat >= 0 else "S"
    utm_zone = "{}{}".format(zone, hemisphere)
    return utm_zone

def loadPoints(truthPath,kmlCorners,checkKml):
    print(truthPath)
    xyzOrg = np.loadtxt(truthPath)
    minXKml = np.min(kmlCorners[:,0])
    maxXKml = np.max(kmlCorners[:,0])
    minYKml = np.min(kmlCorners[:,1])
    maxYKml = np.max(kmlCorners[:,1])
    ret = []
    for i in range(xyzOrg.shape[0]):
        addFlag = False
        x = xyzOrg[i, 0]
        y = xyzOrg[i, 1]
        h = xyzOrg[i, 2]
        if (checkKml):
            if (x >= minXKml and x <= maxXKml and y >= minYKml and y <= maxYKml):
                pt = [x,y]
                if (containsKml(pt, kmlCorners)):
                    addFlag = True
        else:
            addFlag = True

        if (addFlag):
            ret.append([x,y,h])

    return np.array(ret)
    
def registerFast(solutionPoints, truthGrid):
    minError = np.inf
    seen = []
    z1 = np.mean(truthGrid['data'][truthGrid['data'] != INVALID_Z])
    solutionGrid = pointsToGrid(solutionPoints, truthGrid)
    z2 = np.mean(solutionGrid['data'][solutionGrid['data'] != INVALID_Z])
    dz = z2 - z1
    w = truthGrid['w']
    h = truthGrid['h']

    cxG = 0
    cyG = 0
    bestXG = 0
    bestYG = 0
    spanM = 20
    N = 4
    stepG = (int)(spanM / truthGrid['xSpacing'] / N + 0.5)
    maxDG = (int)(spanM / truthGrid['ySpacing'] + 0.5) + 1
    while True:
        for i_dx in range(-N,N+1):
            dx = cxG + i_dx*stepG
            if abs(dx) > maxDG:
                continue
            for j_dy in range(-N,N+1):
                dy = cyG + j_dy * stepG
                if abs(dy) > maxDG:
                    continue
                key = '{},{}'.format(dx,dy)
                if (seen==key):
                    continue
                seen.append(key)

                cnt = 0
                sumErr = 0
                for i in range(0, w, registrationDelta):
                    i2 = i + dx
                    if (i2 < 0 or i2 >= w):
                        continue
                    for j in range(0, h, registrationDelta):
                        j2 = j + dy
                        if (j2 < 0 or j2 >= h):
                            continue
                        z1 = truthGrid['data'][i][j]
                        if (z1 == INVALID_Z):
                            continue
                        z2 = solutionGrid['data'][i2][j2]
                        if (z2 == INVALID_Z):
                            continue
                        err = abs(z2 - dz - z1)
                        sumErr += err * err
                        cnt +=1

                if cnt > 0 :
                    err = math.sqrt(sumErr / cnt) #cnt has not to be 0
                    if (err < minError):
                        minError = err
                        bestXG = dx
                        bestYG = dy

        #print("{} StepG: {}, min error: {}  at ({} , {})".format(datetime.now(), stepG, minError, bestXG, bestYG))
        if (stepG == 1):
            break
        stepG = (int)(stepG * 0.5)
        cxG = bestXG
        cyG = bestYG

    registrationOffset = (bestXG * truthGrid['xSpacing'], bestYG * truthGrid['ySpacing'], dz)

    return registrationOffset    

def pointsToGrid(points, baseGrid, offset=(0, 0, 0)):
    cnt_out = 0
    cnt_over = 0

    r = baseGrid['extent']
    xSpacing = baseGrid['xSpacing']
    ySpacing = baseGrid['ySpacing']
    w = baseGrid['w']
    h = baseGrid['h']

    solutionGrid = {'data':np.full([w,h], INVALID_Z, np.float64)}
    solutionGrid['w'] = w
    solutionGrid['h'] = h
    overlappedList = []
    solutionGrid['extent'] = baseGrid['extent']

    for i in range(points.shape[0]):
        p = points[i]
        x = (int)((p[0] - offset[0] - r['minx']) / xSpacing)
        y = (int)((p[1] - offset[1] - r['miny']) / ySpacing)
        if (x < 0 or y < 0 or x >= w or y >= h):
            cnt_out += 1
            continue
        z = p[2]
        if solutionGrid['data'][x][y] != INVALID_Z :
            cnt_over += 1
        solutionGrid['data'][x][y] = max(z, solutionGrid['data'][x][y])

        if(offset!=(0, 0, 0)): # for check overlapped points
            pp = [x,y,p[0],p[1],p[2]]
            overlappedList.append(pp)


    if(offset!=(0, 0, 0)):
        #np.savetxt("../s2p_output/output_pair/overlappedlist.txt",sorted(overlappedList),fmt="%8d,%8d,%15.4f,%15.4f,%15.4f")
        print('point number out of grid :{}'.format(cnt_out))
        print('point number overlapped in grid :{}'.format(cnt_over))

    return solutionGrid

def readkml(kmlPath):
    with open(kmlPath, 'r') as f:
        a = bs4.BeautifulSoup(f, "lxml").find_all('coordinates')[0].text.split()
    ll_poly = np.array([list(map(float, x.split(','))) for x in a])[:, :2]
    utm_zone = compute_utm_zone(*ll_poly.mean(axis=0))

    zone_number = utm_zone[:-1]
    hemisphere = utm_zone[-1]
    utm_proj = pyproj.Proj(proj='utm', zone=zone_number, ellps='WGS84', datum='WGS84',  south=(hemisphere == 'S'), )
    easting, northing = pyproj.transform(
        pyproj.Proj(init="epsg:4326"), utm_proj, ll_poly[:, 0], ll_poly[:, 1]
    )
    easting = easting.reshape(len(easting), 1)
    northing = northing.reshape(len(northing), 1)
    ret = np.hstack((easting,northing))
    ret = ret[0:4,:]

    if True:
        ret[0][0] = 354052.36522137065
        ret[0][1] = 6182691.101292624
        ret[1][0] = 354755.32500418293
        ret[1][1] = 6182702.11945782
        ret[2][0] = 354765.34005407925
        ret[2][1] = 6182061.65937406
        ret[3][0] = 354062.4287764372
        ret[3][1] = 6182050.640352659

    return ret

def containsKml(pt, kmlCorners):
        result = False
        for i in range(4):
            j = (i + 3) % 4
            y_inFlag = (kmlCorners[i][1] > pt[1]) != (kmlCorners[j][1] > pt[1])
            x_inFlag = pt[0] < (kmlCorners[j][0] - kmlCorners[i][0]) * (pt[1] - kmlCorners[i][1]) / (kmlCorners[j][1] - kmlCorners[i][1]) + kmlCorners[i][0]
            if (y_inFlag and x_inFlag):
                result = not result
        return result
def createTruthGrid(truthPoints):
    list_x = truthPoints[:, 0]
    arrX = np.unique(list_x)
    w = len(arrX)
    r = {'minx': arrX[0]}
    r['maxx'] = arrX[w-1]
    xSpacing = (r['maxx'] - r['minx']) / (w - 1)
    r['minx'] -= xSpacing / 2
    r['maxx'] += xSpacing / 2

    listY = truthPoints[:, 1]
    arrY = np.unique(listY)
    h = len(arrY)
    r['miny'] = arrY[0]
    r['maxy'] = arrY[h-1]
    ySpacing = (r['maxy'] - r['miny']) / (h - 1)
    r['miny'] -= ySpacing / 2
    r['maxy'] += ySpacing / 2

    truthGrid = {'data': np.full([w,h], INVALID_Z, np.float64)}
    truthGrid['extent'] = r
    truthGrid['xSpacing'] = xSpacing
    truthGrid['ySpacing'] = ySpacing
    truthGrid['w'] = w
    truthGrid['h'] = h
    for i in range(truthPoints.shape[0]):
        p = truthPoints[i]
        x = (int)((p[0] - r['minx']) / xSpacing)
        y = (int)((p[1] - r['miny']) / ySpacing)
        if (x < 0 or y < 0 or x > w or y > h):
            continue
        if (x == w): x = w-1
        if (y == h): y = h-1
        truthGrid['data'][x][y] = p[2]

    return truthGrid

def createGrid_height(dispPoints,img_w,img_h):
    # set the sigma_threshold
    sigma_threshold = 1.7
    min_disp = 0
    max_disp = 60
    
    # The empty pixels (holes) are marked by nan values
    truthGrid = np.zeros([img_h,img_w,20])
    pelCounter  = np.zeros([img_h,img_w]).astype('int')
    # Determine the number of points
    numPoints = dispPoints.shape[0]
    
    print(' step 5.2: Putting all points on a grid ...')
    
	# Step 1: Fill in the points using rounding of pixel coordinates
    for i in tqdm(range(numPoints)):
        # Derive the point (x,y) and convert it to int type
        p = np.rint(dispPoints[i,0:2]).astype('int')
        # Get the z point and leave it as integer
        z  = dispPoints[i,2]
        # If the point p points to a valid region
        if (p[1] < img_h) and (p[0] < img_w) and p[1] >= 0 and p[0] >= 0:
            if z >= min_disp and z <= max_disp: # This is a valid pixel
			    # Put the disparity in the next entry in truthGrid
                truthGrid[p[1]][p[0]][pelCounter[p[1]][p[0]]] = z
                # Increment the truthMap which is acting as a counter to the number of 
                # points pointing to this pixel (x,y)
                pelCounter[p[1]][p[0]]  += 1
    print('Height at ({},{}): {}'.format(1088,743,truthGrid[743][1088][0:pelCounter[1088,743]]))
    print('Height at ({},{}): {}'.format(1360,2004,truthGrid[2004][1360][0:pelCounter[2004,1360]]))
    
    # Initialize the disparity map
    disparity = np.zeros([img_h,img_w,])
    
    # Derive the maks
    mask = pelCounter > 0
    
    print(' step 5.3: Compute the first estimate of the disparity map ...')
    
    # Get the list of coordinates to be considered
    coord_list = np.argwhere(mask).tolist()
    
    for coord in tqdm(coord_list):
		# Get the row and column coordinates
        r = coord[0]
        c = coord[1]
		
        # Get the list of possible disparities at (x,y)
        disp = truthGrid[r][c][0:pelCounter[r,c]]
        
        # Compute the standard deviation            
        sigma = np.std(disp)
        if sigma <= sigma_threshold:
            # This can be accurately predicted using the mean
            disparity[r,c] = np.mean(disp)
        else:
            # Take the maximum disparity here			
            disparity[r,c] = np.max(disp)
    return disparity, mask


def get_lon_lat(solutionPoints, aoi):
    # step1.1: get zonestring by using the center of the AOI
    
    # Get the centroid of the area of interest in terms of latitude and longitude
    aoi_lons, aoi_lats = np.asarray(aoi['coordinates'][0][:4]).T
    aoi_lon, aoi_lat = np.mean([aoi_lons, aoi_lats], axis=1)
    
    # Derive the umts zone from the centroid lat and long of the aoi
    zonestring = utils.zonestring_from_lonlat(aoi_lon, aoi_lat) #aoi coordinate

    hOrg = solutionPoints[:,2] #height of xyzOrignal
    print("remove invalid points")
    invalidRows = np.where(hOrg==-9999)
    xyz = np.delete(solutionPoints,invalidRows,axis=0)
    
    # utm to longitude and latitude
    print("Computing the longitude and latitude from utm... ")
    easts = xyz[:,0]
    norths = xyz[:,1]
    heights = xyz[:,2]
    # Derive the longitudes and latitudes
    lons,lats = utils.lonlat_from_utm(easts,norths,zonestring)

    return lons,lats,heights

def affine_param(A, w, h):
    """
    Apply an affine transform to an image.
    This function is from rectification.affine_crop(input_path, A, w, h)
    """

    # determine the rectangle that we need to read in the input image
    output_rectangle = [[0, 0], [w, 0], [w, h], [0, h]]
    x, y, w0, h0 = utils.bounding_box2D(utils.points_apply_homography(np.linalg.inv(A),
                                                                      output_rectangle))
    x, y = np.floor((x, y)).astype(int)
    w0, h0 = np.ceil((w0, h0)).astype(int)

    # compensate the affine transform for the crop
    B = A @ rec.matrix_translation(x, y)

    return B,x, y, w0, h0

def Ori2RecCo(xyExt,S, w, h):
    '''
     # apply the affine transform
    This function is from rectification.affine_crop(input_path, A, w, h) last line
    out = ndimage.affine_transform(aoi.T, np.linalg.inv(B), output_shape=(w, h)).T

    :param xyExt:
    :param S:
    :param w:
    :param h:
    :return:
    '''
    B,x0, y0, w0, h0 = affine_param(S, w, h)
    C = B[:2,:]
    xyExt[:,0] = xyExt[:,0]-x0
    xyExt[:,1] = xyExt[:,1]-y0
    rect_xy = np.dot(xyExt,C.T)
    rect_x0,rect_y0 = np.dot([x0,y0,1],C.T)

    return rect_xy,rect_x0,rect_y0

def s2p_stereo_vision(out_foldername,img_left_filename, img_right_filename,tag,method, mccnn_model_path='None'):
    # import warnings filter
    from warnings import simplefilter
    # ignore all future warnings
    simplefilter(action='ignore', category=FutureWarning)    
    
    # init Paramate
    aoi_challenge = {'coordinates': [[[-58.589468,-34.487061],
                        [-58.581813,-34.487061], [-58.581813,-34.492836],
                        [-58.589468,-34.492836], [-58.589468,-34.487061]]],
       'type': 'Polygon'}
    aoi = aoi_challenge
    	
    # Derive the path of the Challenge.kml file
    roi_kml = os.path.join('../data/mvs_dataset/groundtruth/','Challenge1.kml')
    
    config = {
      "out_dir": out_foldername,
      "images": [
        {"img": img_left_filename},
        {"img": img_right_filename}
      ],
      "roi_kml": roi_kml,
      "horizontal_margin": 20,
      "vertical_margin": 5,
      "tile_size": 300,
      "disp_range_method": "sift",
      "msk_erosion": 0,
      "dsm_resolution":  0.3, # = 0.5  20200304 mChen
      "matching_algorithm": method,
      "temporary_dir": os.path.join(out_foldername,'temp'),
      "mccnn_model_dir": mccnn_model_path,
      'max_processes': None
    }
    # Compute the s2p method to compute the disparity
    s2p.main(config)
    
    # Read data from the kml file
    print('Read data from the kml file')
    kmlCorners = readkml(roi_kml)
    
    # Derive the ground truth path
    #print('Derive the truth path')
    #truthPath = os.path.join('../data/mvs_dataset/groundtruth/','Challenge1_Lidar.xyz')
    # Load the grounth truth points
    #truthPoints = loadPoints(truthPath,kmlCorners,True)
    # Create the grid for ground truth
    #truthGrid = createTruthGrid(truthPoints)

    # Derive the solution path
    print('Derive the solution path')
    solutionPath = os.path.join(out_foldername, 'ply.xyz')
    # Load the solution points
    print('Load the solution points')
    solutionPoints = loadPoints(solutionPath, kmlCorners, False)
    
    # Clean all the folder
    #shutil.rmtree(os.path.join(out_foldername, 'temp'))
    #shutil.rmtree(os.path.join(out_foldername, 'tiles'))
    #os.remove(os.path.join(out_foldername, 'dsm.tif'))
    #os.remove(os.path.join(out_foldername, 'config.json'))
    #os.remove(os.path.join(out_foldername, 'dsm.tif.aux.xml'))
    #os.remove(os.path.join(out_foldername, 'dsm.vrt'))
    #os.remove(os.path.join(out_foldername, 'gdalbuildvrt_input_file_list.txt'))
    #os.remove(os.path.join(out_foldername, 'global_pointing_pair_1.txt'))
    #os.remove(os.path.join(out_foldername, 'ply.xyz'))
    #os.remove(os.path.join(out_foldername, 'tiles.txt'))
    
    out_disp_foldername = os.path.join('./data/',tag)

    # Derive the homographies and additional information that is needed for registration
    metadata_filename = os.path.join(out_disp_foldername, 'metadata.pkl')
    f = open(metadata_filename, 'rb')
    S_left = pickle.load(f)
    S_right = pickle.load(f)
    w = pickle.load(f)
    h = pickle.load(f)
    P_left = pickle.load(f)
    P_right = pickle.load(f)
    f.close()
    
    print('Convert solution points to lon, lat and height ...')
    # Derive the lon lat and heights from file
    lon, lat, height_soln = get_lon_lat(solutionPoints,aoi)
    
    # Get the rpc from the left image
    rpc = utils.rpc_from_geotiff(img_left_filename)
    
    # Project the lon and lat values on the image grid
    x, y = rpc.projection(lon, lat, height_soln)
    xy = np.vstack((x, y)).T
    
    # Initialize a column vector of ones
    xyExt = np.ones(solutionPoints.shape[0]).reshape(-1,1)
    
    # Derive the  points
    Points = np.hstack((xy,xyExt))

    #print(" rect to left image coordinates...")
    rect_xy,_,_ = Ori2RecCo(Points,S_left, w, h)

    # Concatenate the recrified xy coordinates on the left image with the heights
    heightList = np.hstack((rect_xy, height_soln.reshape(-1,1)))
    
    # Derive the height map
    height, mask = createGrid_height(heightList,w,h)

    # Set the disparity that are not defined as nan
    height[~mask] = np.nan

    return height

def get_geotiff_filenames(txt_filename):
    file = open(txt_filename,"r")
    leftImgFile = file.readline() 
    rightImgFile = file.readline() 
    file.close()
    img_left_filename = leftImgFile.split(":",1)[1].split()[0]
    img_right_filename = rightImgFile.split(":",1)[1].split()[0]
    return img_left_filename, 	img_right_filename

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description="compute a disparity map and convert to a height map")
parser.add_argument("--in_foldername", type=str, required=True, help="Input foldername where the disparity map and height map are  stored")
parser.add_argument("--method", type=str, required=True, help="Method used to compute the stereo matching.")
parser.add_argument("--mccnn_model_path", type=str, help="path to the mccnn model.")

def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    # Get the arguments from the parse
    args = parser.parse_args()

    # Use the same disparity map range as done in the xyz2disp.py script
    disp_min = -24
    disp_max =  25
	
    out_foldername = os.path.join('./results/',args.in_foldername,args.method)

    if not os.path.exists(out_foldername):
        os.makedirs(out_foldername)
    
    if args.method == 's2p' or args.method == 's2p-mccnn' or args.method == 's2p2' :
        # Get the name of the filename
        txt_filename = os.path.join('./data/',args.in_foldername,'pair.txt')
        # Get the left and right geotiff filenames
        img_left_filename, 	img_right_filename = get_geotiff_filenames(txt_filename)	
    else:
        # Determine the filenames of the images to be processed
        img_left_filename  = os.path.join('./data/',args.in_foldername,'left.tif')
        img_right_filename = os.path.join('./data/',args.in_foldername,'right.tif') 
    
    if args.method == 'sgbm':
        # Compute stereo vision using sgbm algorithm
        height_map = sgbm_stereo_vision(out_foldername,img_left_filename, img_right_filename,disp_min,disp_max,args.in_foldername)
    elif args.method == 'mccnn':
        # Derive the path to the model
        mccnn_model_path = args.mccnn_model_path
        sub_pixel_interpolation = args.sub_pixel_interpolation

        # Compute stereo vision using sgbm algorithm
        height_map = mccnn_basic_stereo_vision(img_left_filename, img_right_filename,disp_min,disp_max,args.in_foldername,mccnn_model_path,sub_pixel_interpolation)
    elif args.method =='s2p':
        # Compute stereo vision using the s2p framework
        height_map = s2p_stereo_vision(out_foldername,img_left_filename, img_right_filename,args.in_foldername,'sgbm')		
    elif args.method =='s2p-mccnn':
        # Compute stereo vision using the s2p framework
        mccnn_model_path = args.mccnn_model_path
        height_map = s2p_stereo_vision(out_foldername,img_left_filename, img_right_filename,args.in_foldername,'mccnn_basic',mccnn_model_path)		
        
    # Derive the disp filename
    height_filename = os.path.join(out_foldername, 'height.pkl')
        
    file = open(height_filename, 'wb')
 
    # dump information to that file
    pickle.dump(height_map, file)

    # close the file
    file.close()


if __name__ == "__main__":
    main()
