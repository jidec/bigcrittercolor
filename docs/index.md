# Welcome to bigcrittercolor

*bigcrittercolor* is a Python package that provides a framework and pipeline for extracting traits from citizen science images of animals.

It includes functions for **four universal steps** to be applied in sequence to a **bigcrittercolor data folder**.

1. **Image downloading** - from iNaturalist and/or Observation.org 
2. **Segmentation** - extracting insect from background using groundedSAM
3. **Filtering** - filtering out bad (incomplete, overlapping, dead, missing) segments using a human-in-the-loop feature clustering step
4. **Trait extraction** - extracting color traits using thresholding, species-contrastive metric learning using BioEncoder, or color clustering

See the R companion *bigcrittercolorR* for visualization and analysis of extracted traits 

## Installation

* `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118` - download torch and torchvision
* `pip install git+https://github.com/IDEA-Research/GroundingDINO.git@main` - download groundingDINO object detector
* `pip install git+https://github.com/facebookresearch/segment-anything.git@main` - download SegmentAnything
* `pip install git+https://github.com/jidec/bigcrittercolor.git@main` - download bigcrittercolor

## [Google Colab demo](https://colab.research.google.com/drive/1p6D-HTsj33IIrt3pK0HkHOv-DPBcGIqx?usp=sharing)

## Minimal pipeline

    from bigcrittercolor import createBccDataFolder, downloadiNatImagesAndData, inferMasks, filterExtractSegments, extractSimpleColorTraits
	
	createBccDataFolder(parent_folder="C:",new_folder_name="my_bcc_project")
	downloadiNatImagesAndData(taxa_list=["Cylindrophora"],data_folder="C:/my_bcc_project")
	inferMasks(text_prompt="insect", data_folder="C:/my_bcc_project")
	filterExtractSegments(data_folder="C:/my_bcc_project")
	extractSimpleColorTraits(data_folder="C:/my_bcc_project")
	