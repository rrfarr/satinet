#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  data_preparation.py
#  
#  Copyright 2021 reuben
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
#  
import argparse
import urllib.request
import os, ssl
import zipfile
from progressist import ProgressBar
# Specify the parser that will be used to get information from the command line
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description="Script to be used to download and organize the datasets.")
parser.add_argument("--dataset", type=str, required=True, help="The dataset to be downloaded. The supported datasets are i) md for the Middlebury Stereo Vision or ii) IARPA Challenge")

def download_url(url, out_foldername):
    # This is to create an unverified context to download https
    ssl._create_default_https_context = ssl._create_unverified_context   
    
    # Derive the baseline name
    filename = os.path.basename(url) 
    bar = ProgressBar(template="Download |{animation}| {done:B}/{total:B}")
   
    # Download the file  
    urllib.request.urlretrieve(url, filename=os.path.join(out_foldername,filename),reporthook=bar.on_urlretrieve) 
     
def unzip(filename,out_foldername):
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(out_foldername)          
def main(args):
    # Retrieve the arguments inputted from CLI	
    args = parser.parse_args()
    

    if args.dataset == 'md':
        # Make sure there is a folder to contain the data
        out_foldername = './data/'
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
        
        
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
