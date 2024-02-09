# bigcrittercolor
 
bigcrittercolor is a pipeline for automated color trait discovery from mass iNaturalist images

Try the [cloud Colab Demo](https://colab.research.google.com/drive/1p6D-HTsj33IIrt3pK0HkHOv-DPBcGIqx) where in around 10 minutes you can try the package out on your taxon

Installation
1. Install Cuda 11.8
2. Install torch and torchvision (with Cuda 11.8 compatability)
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
3. Install groundingDINO
`pip install git+https://github.com/IDEA-Research/GroundingDINO.git@main
4. Install SegmentAnything
`pip install git+https://github.com/facebookresearch/segment-anything.git@main
5. Install bigcrittercolor
`pip install git+https://github.com/jidec/bigcrittercolor.git@main

Some data paradigms:
1. For simplicity, bigcrittercolor works on a set data folder format that the user must create using createBCCDataFolder() when starting the project. 
    Generally, each data folder should contain observations of one clade in which organism images are similar enough that the same processing arguments work on the whole clade.
    For example, trying to analyze dragonflies and birds in the same folder would be difficult and is not recommended because these images have such different characteristics. 
	Each data folder can usually be construed as a single research project where we want to get color traits for species or individuals of one clade, perhaps just in one location or set of locations or one timespan or set of timespans. 
	The core functions of bigcrittercolor always apply to this data folder, taking a string parameter called "data_folder" as an argument. 
	
2. No comprehensive .csv record is kept that contains info on which images have gone through each processing step, rather:
    - an ID being in masks means it was masked
    - an ID being in segments means it was filtered and segmented
    - an ID being in patterns means it was color clustered
	The only record kept for now is /other/processing_info/failed_mask_infers.txt which contains ids where masking failed so we can skip those in the future

3. Image formats are as follows:
	- all_images - 3-channel RGB .jpgs of variable sizes 
	- masks - (1/3?)-channel binary images (OR normalized but unfiltered 4-channel RGBA segments if the auxiliary segmenter is used in inferMasks())
	- segments - normalized and filtered 4-channel RGBA segments of variable sizes
	- patterns - 4-channel RGBA reductions of continuously shaded segments to discrete, comparable, denoised color patterns 