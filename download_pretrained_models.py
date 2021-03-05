#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import requests
import sys
from zipfile import ZipFile

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def download_file(url,id,destination):
    session = requests.Session()
    response = session.get(url, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)
    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)    
        
    save_response_content(response, destination)    
from google_drive_downloader import GoogleDriveDownloader as gdd

def main(args):
    mccnn_model_path = './Model/MC-CNN/checkpoint/'
    laf_model_path   = './Model/LAFNET/'
	
    if not os.path.exists(mccnn_model_path):
        print('Downloading the MC-CNN pretrained model...')
        # Download and unzip the file
        gdd.download_file_from_google_drive(\
            file_id='1h2KuRM9YXvcPSdS8VVJ2Fbi7HCRo2hjc' , \
            dest_path='./Model/MC-CNN.zip', unzip=True)
        # Delete the zip file
        os.remove('./Model/MC-CNN.zip')
    else:
        print('MC-CNN Model already exists!')
        
    if not os.path.exists(laf_model_path):    
        print('Downloading the LAFNET pretrained model...')
        # Download and unzip the file
        gdd.download_file_from_google_drive(\
            file_id='1pIeSKet2e9qmdZ2zeyvH56lPri6UQrh-' , \
            dest_path='./Model/LAFMET.zip', unzip=True)
        # Delete the zip file
        os.remove('./Model/LAFMET.zip')
    else:
        print('LAFNET Model already exists!')

if __name__ == '__main__':
	
    sys.exit(main(sys.argv))
