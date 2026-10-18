# Datasets

This repository contains code only. No data is included.

## External validation dataset (sample)

Collected with five identical Raspberry Pi 4B prototype units, one per vehicle, mounted on the rear-view mirror. 
**Sample download:** [Google Drive](https://drive.google.com/file/d/1VaxI7T54LuosRZxvCoqnssUc3q3slA_e/view?usp=sharing)

The full external dataset is available from the corresponding author on reasonable request.

## RoadSens-4M

Used for federated training and internal testing. 

Access and terms are described in the dataset publication:

> A. Khandakar et al., "RoadSens-4M: A Multimodal Smartphone & Camera Dataset for Holistic Road-Way Analysis," *Scientific Data*, 2026. Figshare Link: https://figshare.com/articles/dataset/RoadSens-4M_Dataset/30341143

## Model input

GraphEffiNet expects RGB frames resized to 224 × 224 (Lanczos resampling in the paper) and normalized with ImageNet statistics:

- mean = (0.485, 0.456, 0.406)
- std = (0.229, 0.224, 0.225)

Class indices: 0 = normal, 1 = bump, 2 = pothole.

## Contact

Dr. Amith Khandakar, Qatar University, amitk@qu.edu.qa
