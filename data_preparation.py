#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  data_preparation.py
#  
#  Copyright 2021 Reuben Farrugia
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  This script can be used to download the Middlebury Stereo Vision Dataset
#  and the IARPA Multi-View Stereo 3D Mapping Challenge Dataset. This script
#  takes two inputs where only one is mandatory
#
#  -d (or --dataset): specifies the dataset that will be downloaded. the script 
#                     supports two options: i) md which will download the Middlebury 
#                     Stereo Vision Dataset and ii) iarpa which will download the 
#                     IARPA Multi-View Stereo 3D Mapping Challenge Dataset.
#  -l (or --img_list): specifies a list of images that will be downloaded. This option
#                     is useful when downloading a list of images from the IARPA 
#                     Multi-View Stereo 3D Mapping Challenge Dataset.
#  
#  Example
#  -------
#
#  python3 data_preparation.py -d md
#  python data_preparation.py -d iarpa -l 18DEC15WV031000015DEC18140522-P1BS-500515572020_01_P001_________AAE_0AAAAABPABJ0.TIF 18DEC15WV031000015DEC18140544-P1BS-500515572060_01_P001_________AAE_0AAAAABPABJ0.TIF```



import argparse
import urllib.request
import os, ssl
import zipfile
from progressist import ProgressBar

# Specify the parser that will be used to get information from the command line
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description="Script to be used to download and organize the datasets.")
parser.add_argument('-d', "--dataset", type=str, required=True, help="The dataset to be downloaded. The supported datasets are i) md for the Middlebury Stereo Vision or ii) iarpa for the IARPA Challenge")
parser.add_argument('-l','--img_list', nargs='+',help='List of images to be downloaded from the iarpa', required=False) 

#########################################################################################################
# Script used to download a file from a specific url
#########################################################################################################
# input:  url specifying the full url of the file to be downloaded
#         out_foldername specifies the output foldername
def download_url(url, out_foldername):
    # This is to create an unverified context to download https
    ssl._create_default_https_context = ssl._create_unverified_context   
    
    # Derive the baseline name
    filename = os.path.basename(url) 
    bar = ProgressBar(template="Download |{animation}| {done:B}/{total:B}")
   
    # Download the file  
    urllib.request.urlretrieve(url, filename=os.path.join(out_foldername,filename),reporthook=bar.on_urlretrieve) 
#########################################################################################################
# Script to unzip the file
#########################################################################################################
# input: filename of the zipfile to be unpacked
#        out_foldername specifies the output foldername
def unzip(filename,out_foldername):
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(out_foldername)          
def main(args):
    # Retrieve the arguments inputted from CLI	
    args = parser.parse_args()
   
    if args.dataset == 'md':
        print("Downloading the MiddEval3 dataset")

        # Make sure there is a folder to contain the data
        out_foldername = './Data/'
        if not os.path.exists(out_foldername):
            os.makedirs(out_foldername)	
        #########################################################################################################
        # DOWNLOADING THE MIDDEVAL3 DATASET
        #########################################################################################################
        
        # Download the MiddEval3-data-H.zip file
        print('Downloading the MiddEval3-data-H.zip file')
        download_url("https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-data-H.zip",out_foldername)
        
        # Download the MiddEval3-GT0-H.zip file
        print('Downloading the MiddEval3-GT0-H.zip file')
        download_url('https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-GT0-H.zip',out_foldername)
        
        # Download the MiddEval3-GT1-H.zip file
        print('Downloading the MiddEval3-GT1-H.zip file')
        download_url('https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-GT1-H.zip',out_foldername)

        #########################################################################################################
        # UNZIP THE THREE DOWNLOADED FILES
        #########################################################################################################
        print('Unzipping all the files in the dataset ...')
        unzip(os.path.join(out_foldername,'MiddEval3-data-H.zip'), out_foldername)
        unzip(os.path.join(out_foldername,'MiddEval3-GT0-H.zip'), out_foldername)
        unzip(os.path.join(out_foldername,'MiddEval3-GT1-H.zip'), out_foldername)
 
        #########################################################################################################
        # CLEAN UP THE DATA
        #########################################################################################################
        print('Delete the zip files ...')
        os.remove(os.path.join(out_foldername,'MiddEval3-data-H.zip'))
        os.remove(os.path.join(out_foldername,'MiddEval3-GT0-H.zip'))
        os.remove(os.path.join(out_foldername,'MiddEval3-GT1-H.zip'))
    elif args.dataset == 'iarpa':
        print("Downloading the IARPA dataset")
        # url where the iarpa data is stored
        url_data = 'http://menthe.ovh.hw.ipol.im/IARPA_data/cloud_optimized_geotif/'
        # Make sure there is a folder to contain the data
        out_foldername = './Data/iarpa/'
        if not os.path.exists(out_foldername):
            os.makedirs(out_foldername)	
        # Determine the number of images to be downloaded
        for img_filename in args.img_list:
            # Derive the path where we are going to save the data
            out_filename = os.path.join(out_foldername, img_filename)
            
            if not os.path.exists(out_filename):
                # Derive the url to be downloaded
                url = os.path.join(url_data, img_filename)
                # The file does not exist so we have to download it
                print('Downloading file {}'.format(img_filename))
                # Download the image
                download_url(url, out_foldername)
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
