# Stereo Matching

This script will use the [S2P processing pipeline](https://github.com/MISS3D/s2p) developed by CNES which rectifies, matches and derives the 
Digital Elevation Model (DEM) from two or more satellite images.
One can run the matching script using the following command

```console
./matching.py  --in_foldername ./Data/Iarpa --method s2p-mccnn --mccnn_model_path ./Model/MC-CNN/
```

This script takes three command line inputs. The *in_foldername* argument specifies the foldername where the GEOTIFF images are stored.
The *method* argument specifies the stereo-matching method used.

