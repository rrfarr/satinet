# Stereo Matching

This script will use the [S2P processing pipeline](https://github.com/MISS3D/s2p) developed by CNES which rectifies, matches and derives the 
Digital Elevation Model (DEM) from two or more satellite images.
One can run the matching script using the following command

```console
./mccnn-train.py -m Model/MC-CNN/ -d ./Data/MiddEval3/trainingH/
```

The attribute **m** specifies the folder where the model will be stored while the attribute
**d** specifies the directory containing the training data.
The list of training files is loaded from the md_list.json file.

