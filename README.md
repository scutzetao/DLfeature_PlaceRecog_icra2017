# DLfeature_PlaceRecog_icra2017

This repository provides scripts to extract features using models from the paper "Deep learning features at scale for visual place recognition" published by Zetao Chen, et al. on ICRA 2017. 

Steps:
1) Please install caffe packages by following the link: http://caffe.berkeleyvision.org/install_apt.html 

2) Download the "HybridNet" model from https://goo.gl/kF6nQh and copy them to the folder "HybridNet" in this repository;

3) You will need to update the file 'nordland.txt' using the file pathes of your images. Each line in this file specifies one image whose feature is to be extracted. 

4) The file 'extract_feat_usingAMOS.py' extract features from the fc7 layer. Update it if you need to extract features from other layers.

5) Run the 'extract_feat.sh'


