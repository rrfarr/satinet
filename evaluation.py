#!/usr/bin/env python
# -*- coding: utf-8 -*-

import bs4
import numpy as np
import math
from datetime import datetime
import matplotlib.pyplot as plt
import utm
import pyproj
import os.path
from collections import Counter
import argparse

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

def utm_proj(utm_zone):
    """
    Return a pyproj.Proj object that corresponds
    to the given utm_zone string

    Args:
        utm_zone (str): UTM zone number + hemisphere (eg: '30N')

    Returns:
        pyproj.Proj: object that can be used to transform coordinates
    """
    zone_number = utm_zone[:-1]
    hemisphere = utm_zone[-1]
    return pyproj.Proj(
        proj='utm',
        zone=zone_number,
        ellps='WGS84',
        datum='WGS84',
        south=(hemisphere == 'S'),
    )

def lonlat_to_utm(lon, lat, utm_zone):
    """
    Compute UTM easting and northing of a given lon, lat point.

    Args:
        lon (float): longitude
        lat (float): latitude
        utm_zone (str): UTM zone, e.g. "14N" or "14S"

    Returns:
        easting, northing
    """
    e, n = pyproj.transform(pyproj.Proj(init="epsg:4326"), utm_proj(utm_zone),
                            lon, lat)
    return e, n


def utm_from_latlon(lats, lons):
    n = utm.latlon_to_zone_number(lats[0], lons[0])
    l = utm.latitude_to_zone_letter(lats[0])
    proj_src = pyproj.Proj('+proj=latlong')
    proj_dst = pyproj.Proj('+proj=utm +zone={}{}'.format(n, l))
    return pyproj.transform(proj_src, proj_dst, lons, lats)

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

def postprocessSolutionGrid(solutionGrid):
    postNum = 0
    W = solutionGrid['w']
    H = solutionGrid['h']
    solutionPostImg = solutionGrid['data'].copy()
    while True:
        for i in range(W):
            for j in range(H):
                if solutionPostImg[i][j] == INVALID_Z:
                    z8=[]
                    for dx in (-1, 0, 1):
                        x = i + dx
                        if (x < 0 or x >= W):
                            continue
                        for dy in (-1, 0, 1):
                            y = j + dy
                            if (y < 0 or y >= H):
                                continue
                            if (solutionPostImg[x][y] == INVALID_Z):
                                continue
                            z8.append(solutionPostImg[x][y])

                    z8 = sorted(z8)
                    z8num = len(z8)
                    if z8num > 0 :
                        solutionPostImg[i][j] = np.percentile(z8,50)
                        postNum += 1

        if sum(solutionPostImg[solutionPostImg==INVALID_Z]) == 0:
            break

    print('postprocess new points:{}'.format(postNum))
    return solutionPostImg

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

def completeness(truthGrid,solutionGrid,dz):
    cnt = 0
    goodReal = 0
    goodExt = 0
    badReal = 0
    badTruth = 0
    badExt = 0
    completenessImg = np.full((truthGrid['w'],truthGrid['h']),15)
    for i in range(truthGrid['w']):
        for j in range(truthGrid['h']):
            z1 = truthGrid['data'][i][j]
            z2 = solutionGrid['data'][i][j]
            if (z2 != INVALID_Z):
                if (z1 != INVALID_Z):
                    cnt += 1
                    if (abs(z2 - dz - z1) < COMPLETENESS_THRESHOLD):
                        goodReal += 1
                        completenessImg[i][j] = 45
                    else:
                        badReal += 1
                else:
                    badTruth += 1
            else:
                if (z1 != INVALID_Z):
                    cnt += 1
                    breakFlag = False
                    for dx in (-1, 0, 1):
                        x = i + dx
                        if (x < 0 or x >= truthGrid['w']):
                            continue
                        for dy in (-1, 0, 1):
                            y = j + dy
                            if (y < 0 or y >= truthGrid['h']):
                                continue
                            z8 = solutionGrid['data'][x][y]
                            if (z8 == INVALID_Z):
                                continue
                            if (abs(z8 - dz - z1) < COMPLETENESS_THRESHOLD):
                                goodExt += 1
                                completenessImg[i][j] = 30
                                breakFlag = True
                                break
                            else:
                                badExt += 1
                        if breakFlag:
                            break

    if (cnt > 0):
        good = goodReal + goodExt
        completeVal = good / cnt
        print('valid points number in solution grid: {}'.format(goodReal))
        print('solution points number bigger than thread: {}'.format(badReal))
        print('solution points number with invalid truth: {}'.format(badTruth))
        print('valid neighbor points number in solution grid: {}'.format(goodExt))
        print('invalid neiborgh points number of solution grid: {}'.format(badExt))
        print('total valid points number in solution grid: {}'.format(good))
        print('total valid points number in truth grid: {}'.format(cnt))
        print('Completeness:{}'.format(completeVal))

        return completeVal,completenessImg
    else:
        return 0,completenessImg

def rmse(truthGrid, grid,dz):
    cnt = 0
    err = 0
    for i  in range(truthGrid['w']):
        for j in range(truthGrid['h']):
            z1 = truthGrid['data'][i][j]
            if (z1 == INVALID_Z):
                continue
            z2 = grid['data'][i][j]
            if (z2 == INVALID_Z):
                continue
            diff = z2 - dz - z1
            cnt += 1
            err += diff * diff
    if (cnt > 0):
        accurateVal = math.sqrt(err / cnt)
        print('both truth and solution valid point number:{}'.format(cnt))
        print('Squared difference:{}'.format(err))
        print('RMSE:{}'.format(accurateVal))

        return accurateVal
    else:
        return 0

def medianZDiff(truthGrid, grid, dz, absFlag=True):
    cnt = 0
    errBuffer = []
    w = truthGrid['w']
    h = truthGrid['h']
    for i in range(0, w, 1): #registrationDelta):
        for j in range(0, h, 1): #registrationDelta):
            z2 = grid['data'][i][j]
            if (z2 == INVALID_Z):
                continue
            z1 = truthGrid['data'][i][j]
            if (z1 == INVALID_Z):
                continue
            diff = z2 - dz - z1
            if (absFlag):
                diff = abs(diff)
            errBuffer.append(diff)
            cnt += 1

    if (cnt > 0):
        errList = np.unique(errBuffer)
        medianVal = np.percentile(errList,50)
        print('both truth and solution valid point number:{}'.format(cnt))
        print('Minimum error:{}'.format(errList[0]))
        print('median error:{}'.format(medianVal))
        print('Maximum error:{}'.format(errList[len(errList) - 1]))

        return medianVal
    else:
        return 0

def score(truthGrid,solutionGrid,dz):
    ret = {'rmse':rmse(truthGrid, solutionGrid, dz)}
    ret['medianZDiff'] = medianZDiff(truthGrid, solutionGrid, dz);
    ret['completeness'],completenessImg = completeness(truthGrid,solutionGrid,dz)

    return ret, completenessImg

def Grid4Shwo(Grid,maxVal = 50):
    Grid['data'][Grid['data']==INVALID_Z] = maxVal
    Grid['minVal'] = np.nanmin(Grid['data'])
    Grid['maxVal'] = np.nanmax(Grid['data'])

def showGrid(truthGrid,solutionPostImg,dz,completenessImg,completeness,rms,median,foldername):
    Grid4Shwo(truthGrid)

    maxVal = 50
    minVal = truthGrid['minVal']

    truthGrid['data'][truthGrid['data'] < 0] = 0
    solutionPostImg = solutionPostImg - dz
    solutionPostImg[solutionPostImg < 0] = 0
    truthGrid['data'][truthGrid['data'] > maxVal] = maxVal
    solutionPostImg[solutionPostImg > maxVal] = maxVal

    truthGrid['data'][0][1] = 0
    solutionPostImg[0][1] = 0
    truthGrid['data'][0][0] = maxVal
    solutionPostImg[0][0] = maxVal
    completenessImg[0][1] = 0
    completenessImg[0][0] = maxVal

    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.text(.25, .25, 'good:{} neighbour:{} othres:{}'.format(completeness,rms,median))
    plt.subplot(2,2,1)
    plt.imshow(truthGrid['data'].T , 'jet'), plt.axis('off')
    plt.title('Ground Truth')
    #plt.colorbar()
    plt.subplot(2,2,2)
    plt.imshow(solutionPostImg.T , 'jet'), plt.axis('off')
    #plt.title('Solution:{}'.format(solutionFile))
    #plt.colorbar()
    
    # Save the ground truth and the solution grid    
    np.save(os.path.join(foldername,'truthGrid.npy'), truthGrid['data'].T)
    np.save(os.path.join(foldername,'solutionPostImg.npy'), solutionPostImg.T)

    diffGrid = abs(solutionPostImg - truthGrid['data']).astype(np.int)
    maxDiff = 6
    diffGrid[diffGrid > maxDiff] = maxDiff
    plt.subplot(2,2,3)
    plt.imshow(diffGrid.T , 'jet'), plt.axis('off')
    plt.title('Solution - Truth')
    plt.colorbar()

    if False:
        diffGrid[diffGrid > 0] = 30
        diffGrid[diffGrid < 0] = 10
        diffGrid[diffGrid == 0] = 20
        diffGrid[0][1] = 0
        diffGrid[0][0] = maxVal

        plt.subplot(2,2,4)
        plt.imshow(completenessImg.T , 'jet'), plt.axis('off')
        plt.title('Complete')
        plt.colorbar()

    else:

        sortDispCount = maxDiff
        Hist = np.zeros([sortDispCount, 1])
        i = 0
        for d in range(maxDiff):
            diffCont = Counter(diffGrid[diffGrid == d])
            Hist[i][0] = diffCont[d]
            i += 1

        plt.subplot(2, 2, 4)
        plt.bar(range(maxDiff), Hist[:maxDiff, 0])
        plt.xticks(range(maxDiff), rotation=45)

    #plt.show()

#solutionPath = './data/Challenge1_Lidar.xyz'
#solutionFile = 'ply.xyz'
#solutionPath = os.path.join('../s2p_output/output_pair/',solutionFile)
fast_registration = True
registration_method = 'rmse'

registrationDelta = 5
zGraphSpan = -1
INVALID_Z = -9999
COMPLETENESS_THRESHOLD = 1
DEFAULT_SPACING = 0.3

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description="analyse the output of the stereo matching process")
parser.add_argument('-k', "--kmlPath", type=str, required=True, help="Directory where the images to be processed for stereo vision are stored")
parser.add_argument('-t', "--truthPath", type=str, required=True, help="Directory where the images to be processed for stereo vision are stored")
parser.add_argument('-r', "--resultsPath", type=str, required=True, help="Directory where the images to be processed for stereo vision are stored")
parser.add_argument('-o',"--out_subdirs", type=str, required=False, help="Directory where the model will be stored")

# default="../data/mvs_dataset/groundtruth/Challenge1_Lidar.xyz"
def main():
    args = parser.parse_args()

    solutionPath = os.path.join(args.resultsPath, 'ply.xyz')
    out_foldername = os.path.join(args.resultsPath)

    # Get only the folder path
    out_foldername = out_foldername.replace('results','analysis')

    if not os.path.exists(out_foldername):
        os.makedirs(out_foldername)
   
    # Derive the output filename
    out_filename = os.path.join(out_foldername,'summary.txt') 
    
    file1 = open(out_filename,"w") 
    # Derive the out_dir
    out_dir = args.resultsPath

    print('{} readkml ...'.format(datetime.now()))
    file1.write('{} readkml ...\n'.format(datetime.now()))
    kmlCorners = readkml(args.kmlPath)

    print('{} loading truth data ...'.format(datetime.now()))
    file1.write('{} loading truth data ...\n'.format(datetime.now()))
    
    truthPoints = loadPoints(args.truthPath,kmlCorners,True)
    # Calculate the number of truth points
    ntruth_pts = truthPoints.shape[0]
    print('truthPoints:{}'.format(ntruth_pts))
    file1.write('truthPoints:{}\n'.format(ntruth_pts))
    
    
    
    # Determine the number of 
    ntruth_valid_pts = sum(truthPoints != INVALID_Z)[2]
    print('valid truth Points Number:{}'.format(ntruth_valid_pts))
    file1.write('valid truth Points Number:{}\n'.format(ntruth_valid_pts))
    rate_valid_truth = ntruth_valid_pts/ntruth_pts
    print('rate valid truth points: {}'.format(rate_valid_truth))
    file1.write('rate valid truth points: {}\n'.format(rate_valid_truth))


    ## CREATE TRUTH GRID
    print('{} creating Truth Grid ...'.format(datetime.now()))
    file1.write('{} creating Truth Grid ...\n'.format(datetime.now()))
    truthGrid = createTruthGrid(truthPoints)
    print('truth Grid scale: {} * {} = {}'.format(truthGrid['data'].shape[0],truthGrid['data'].shape[1],truthGrid['data'].shape[0]*truthGrid['data'].shape[1]))
    print('valid truth Points Number:{}'.format(sum(sum(truthGrid['data'] != INVALID_Z))))
    
    file1.write('truth Grid scale: {} * {} = {}\n'.format(truthGrid['data'].shape[0],truthGrid['data'].shape[1],truthGrid['data'].shape[0]*truthGrid['data'].shape[1]))
    file1.write('valid truth Points Number:{}\n'.format(sum(sum(truthGrid['data'] != INVALID_Z))))

    ## LOADING SOLUTION DATA
    # Derive the solution path
    
    print('{} loading solution data ...'.format(datetime.now()))
    file1.write('{} loading solution data ...\n'.format(datetime.now()))
    solutionPoints = loadPoints(solutionPath, kmlCorners, False)
    print('solution Points Number:{}'.format(solutionPoints.shape[0]))
    print('valid solution Points Number:{}'.format(sum(solutionPoints != INVALID_Z)[2]))
    
    file1.write('solution Points Number:{}\n'.format(solutionPoints.shape[0]))
    file1.write('valid solution Points Number:{}\n'.format(sum(solutionPoints != INVALID_Z)[2]))

    print('{} registering ...'.format(datetime.now()))
    file1.write('{} registering ...\n'.format(datetime.now()))
    
    registrationOffset = (0.0, -1.5000000596046448, 0.8859839707876667)
    #registrationOffset = registerFast(solutionPoints, truthGrid)
    print('{} register to {}'.format(datetime.now(),registrationOffset))
    file1.write('{} register to {}\n'.format(datetime.now(),registrationOffset))

    print('{} creating solution Grid ...'.format(datetime.now()))
    file1.write('{} creating solution Grid ...\n'.format(datetime.now()))
    solutionGrid = pointsToGrid(solutionPoints, truthGrid, registrationOffset)
    print('solution Grid scale: {} * {} = {}'.format(solutionGrid['data'].shape[0],solutionGrid['data'].shape[1],solutionGrid['data'].shape[0]*solutionGrid['data'].shape[1]))
    print('valid solution Points Number:{}'.format(sum(sum(solutionGrid['data'] != INVALID_Z))))
    file1.write('solution Grid scale: {} * {} = {}\n'.format(solutionGrid['data'].shape[0],solutionGrid['data'].shape[1],solutionGrid['data'].shape[0]*solutionGrid['data'].shape[1]))
    file1.write('valid solution Points Number:{}\n'.format(sum(sum(solutionGrid['data'] != INVALID_Z))))
    file1.write('non valid solution Points Number:{}\n'.format(sum(sum(solutionGrid['data'] == INVALID_Z))))

    solutionPostImg = postprocessSolutionGrid(solutionGrid)
    solutionGrid['data'] = solutionPostImg

    print('{} scoring ...'.format(datetime.now()))
    file1.write('{} scoring ...\n'.format(datetime.now()))
    result, completenessImg = score(truthGrid,solutionGrid,registrationOffset[2])
    print('Completenese:{} \n RMSE:{} \n MedianE:{}'.format(result['completeness'],result['rmse'],result['medianZDiff']))
    file1.write('Completenese:{} \n RMSE:{} \n MedianE:{}\n'.format(result['completeness'],result['rmse'],result['medianZDiff']))

    gdcnt = sum(sum(completenessImg==45))
    neicnt = sum(sum(completenessImg==30))
    othcnt = sum(sum(completenessImg==15))
    print('good:{} neighbour:{} othres:{}'.format(gdcnt,neicnt,othcnt))
    file1.write('good:{} neighbour:{} othres:{}\n'.format(gdcnt,neicnt,othcnt))
    file1.close() 
    showGrid(truthGrid,solutionPostImg,registrationOffset[2],completenessImg,result['completeness'],result['rmse'],result['medianZDiff'],out_foldername)
    
if __name__ == "__main__":
    main()
    print('{} The End !'.format(datetime.now()))
