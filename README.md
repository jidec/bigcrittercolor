# bigcrittercolor
 
Data paradigms:
1. For simplicity, bigcrittercolor works on a set data folder format that the user must create using createBCCDataFolder() when starting the project. 
    Generally, each data folder should contain observations of one clade in which organism images are similar enough that the same processing arguments work on the whole clade.
    For example, trying to analyze dragonflies and birds in the same folder would be very difficult because these images have such different characteristics. 
	Each data folder can usually be construed as a single research project where we want to get color traits for species or individuals of one clade, perhaps just in one location or set of locations or one timespan or set of timespans. 
	The core functions of bigcrittercolor always apply to this data folder. 
	
2. No .csv record is kept that contains info on which images have gone through each processing step, rather:
    - an ID being in masks means it was masked
    - an ID being in segments means it was filtered and segmented
    - an ID being in patterns means it was color clustered

2. Image formats are as follows:
	- all_images - 3-channel RGB .jpgs of variable sizes 
	- masks - (1/3?)-channel binary images (OR normalized but unfiltered 4-channel RGBA segments if the auxiliary segmenter is used in inferMasks())
	- segments - normalized and filtered 4-channel RGBA segments of variable sizes
	- patterns - 4-channel RGBA reductions of continuously shaded segments to discrete, comparable, denoised color patterns 