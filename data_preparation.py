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

# Specify the parser that will be used to get information from the command line
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description="Script to be used to download and organize the datasets.")
parser.add_argument("--dataset", type=str, required=True, help="The dataset to be downloaded. The supported datasets are i) md for the Middlebury Stereo Vision or ii) IARPA Challenge")

            
def main(args):
    # Retrieve the arguments inputted from CLI	
    args = parser.parse_args()
    
    # This is to create an unverified context to download https
    ssl._create_default_https_context = ssl._create_unverified_context    

    if args.dataset == 'md':
        # Make sure there is a folder to contain the data
        out_foldername = './data/MiddEval3/'
        if not os.path.exists(out_foldername):
            os.makedirs(out_foldername)		
        urllib.request.urlretrieve("https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-data-H.zip", filename=os.path.join(out_foldername,"MiddEval3-data-H.zip")) 
	
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
