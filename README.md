# zebrafish_image_segmentation

**Modified UNet for Zebrafish head and operculum segmentation**

**_Description_**
- Test Images folder contains the test images, you may use 
- Models folder contains the trained models directly used for testing
- Model architecture is defined in unet_modified.py in src folder
- Main file is run.py in src folder
- All the supporting functions are written in src/utils.py
- Custom _loss functions_ used in the models are written in src/loss_functions.py for **compiling the model**
- Code is compatible with the open-source tool Cytomine ULiège R&D version (https://uliege.cytomine.org)

**_Reference_**
- Deep Learning appraoches for Head and Operculum Segmentation in Zebrafish Microscopy Images, Navdeep Kumar et. al. In the 19th International Conference on Computer Analysis of Images and Patterns (CAIP-2021)

