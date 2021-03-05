#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
import LibMccnn
import matplotlib.pyplot as plt
import tensorflow as tf

from libs2p import utils as utils
from libs2p import pointing_accuracy as pointing_accuracy

############################################################################################################################
# S2P Processing Pipeline: This script is used to call the S2P Processing pipeline
############################################################################################################################
def s2p_stereo_vision(out_foldername,img_left_filename, img_right_filename,data_foldername,method, mccnn_model_path='None',laf_model_path='None'):
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
    roi_kml = os.path.join(data_foldername, 'groundtruth','Challenge1.kml')
    
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
      'laf_model_dir': laf_model_path,
      'max_processes': None
    }
    # Compute the s2p method to compute the disparity
    s2p.main(config)

# Parsers for command line
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description="compute a disparity map and convert to a height map")
parser.add_argument("--in_foldername", type=str, required=True, help="Input foldername where the disparity map and height map are  stored")
parser.add_argument("--method", type=str, required=True, help="Method used to compute the stereo matching. s2p or s2p-mccnn")
# Note: s2p-mccnn also incorporates the median filtering of the cost volume - so this is effectively S2P-MCCNN-Filt as per publication
parser.add_argument("--mccnn_model_path", type=str, help="path to the mccnn model.")
parser.add_argument("--laf_model_path", type=str, help="path to the lafnet model.")

def main():
    #CUDA_VISIBLE_DEVICES=""
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    
    # Get the arguments from the parse
    args = parser.parse_args()
    if args.method == 's2p-mccnn' and args.mccnn_model_path is None:
        parser.error("The path to the model needs to be defined using --mccnn_model_path")
	
    out_foldername = os.path.join('./Results/',args.method.upper())

    if not os.path.exists(out_foldername):
        os.makedirs(out_foldername)
        
    # Specify the left and right images to be considered for stereo matching    
    img_left_filename = os.path.join(args.in_foldername,'18DEC15WV031000015DEC18140522-P1BS-500515572020_01_P001_________AAE_0AAAAABPABJ0.TIF')
    img_right_filename = os.path.join(args.in_foldername,'18DEC15WV031000015DEC18140544-P1BS-500515572060_01_P001_________AAE_0AAAAABPABJ0.TIF')
    
    if args.method =='s2p':
        # Compute stereo vision using the s2p framework
        # height_map = s2p_stereo_vision(out_foldername,img_left_filename, img_right_filename,args.in_foldername,'sgbm')		
        s2p_stereo_vision(out_foldername,img_left_filename, img_right_filename,args.in_foldername,'sgbm')		
    elif args.method =='s2p-mccnn':
        # Compute stereo vision using the s2p framework
        mccnn_model_path = args.mccnn_model_path
        #height_map = s2p_stereo_vision(out_foldername,img_left_filename, img_right_filename,args.in_foldername,'mccnn_basic',mccnn_model_path)		
        s2p_stereo_vision(out_foldername,img_left_filename, img_right_filename,args.in_foldername,'mccnn_basic',mccnn_model_path)		
    elif args.method == 's2p-mccnn-laf':
        # Compute the stereo vision
        mccnn_model_path = args.mccnn_model_path
        laf_model_path =  args.laf_model_path
        s2p_stereo_vision(out_foldername,img_left_filename, img_right_filename,args.in_foldername, 'mccnn_laf',mccnn_model_path,laf_model_path)

if __name__ == "__main__":
    main()
