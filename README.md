# Lesion-detection-in-mammography-images

# Getting Started
* run `python get_sample.py` to generatetxt files
* run `aria2c -j 6 -x 8 -s 8 -c -i <filename>.txt --load-cookies=cookies.txt` to download the training data in bulk
* run `pyton preprocess_sample.py` to convert dicom images into png and create labels for lesion boxes
