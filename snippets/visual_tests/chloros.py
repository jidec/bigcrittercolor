from bigcrittercolor import createBccDataFolder
from bigcrittercolor import downloadiNatImagesAndData
from bigcrittercolor import inferMasks
from bigcrittercolor import filterExtractSegments
from bigcrittercolor import clusterColorsToPatterns

#createBccDataFolder("D:/bcc","chloros")

genera = [
    #"Africocypha",
    #"Aristocypha",
    #"Calocypha",
    #"Chlorocypha",
    #"Cyrano",
    #"Disparocypha",
    #"Heliocypha",
    #"Heterocypha",
    #"Indocypha",
    #"Libellago",
    #"Melanocypha",
    "Pachycypha",
    "Platycypha",
    "Rhinocypha",
    "Rhinoneura",
    "Stenocypha",
    "Sundacypha"
]
#downloadiNatImagesAndData(taxa_list=genera,download_records=False,download_images=True,data_folder="D:/bcc/chloros")
#aux_seg_location = "D:/bcc/damsels_segmenter/aux_segmenter_unetpp.pt"
#aux_seg_location = "D:/bcc/dragonfly_segmenter/aux_segmenter_unetpp_dragonflies.pt"
#inferMasks(skip_existing=True,sam_location="D:/bcc/sam.pth",aux_segmodel_location=aux_seg_location,
#          data_folder="D:/bcc/chloros",show_indv=False)

#filterExtractSegments(data_folder="D:/bcc/chloros",used_aux_segmodel=True)

# other imports
from multiprocessing import freeze_support, set_start_method
from bigcrittercolor.helpers import _getBCCIDs
from bigcrittercolor import writeBasicColorMetrics

def main():
    #ids = _getBCCIDs(type="segment", data_folder="D:/bcc/chloros")
    # whatever you previously ran at top-level
    #clusterColorsToPatterns(img_ids=ids,data_folder="D:/bcc/chloros",n_processes=3,
    #                        cluster_args={'n':10, 'algo':"gaussian_mixture",'unique_values_only':True},
    #                        by_patches=False,show_indv=False,group_cluster_records_colname="species",
    #                        write_subfolder="species")
    #clusterColorsToPatterns(img_ids=ids,data_folder="D:/bcc/chloros",n_processes=3,group_cluster_records_colname=None,
    #                        colorspace="hls", blur_args=None,
    #                       cluster_args={'n':10, 'algo':"gaussian_mixture",'unique_values_only':True},
    #                       by_patches=False,show_indv=False,preclustered=True,preclust_read_subfolder="species")
    writeBasicColorMetrics(from_stage="pattern",data_folder="D:/bcc/chloros")

if __name__ == '__main__':
    # safe on Windows; harmless otherwise
    freeze_support()

    # optional: ensure spawn start method; ignore if already set
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    main()