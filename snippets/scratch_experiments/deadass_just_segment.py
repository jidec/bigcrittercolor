from bigcrittercolor import inferMasks
from bigcrittercolor import writeBasicColorMetrics

#inferMasks(data_folder="D:/bcc/new_random_dragonflies2",aux_segmodel_location="D:/bcc/dragonfly_segmenter/aux_segmenter_unetpp_dragonflies.pt",
#           sam_location="D:/bcc/sam.pth",skip_existing=False,show=True,show_indv=True)

writeBasicColorMetrics(img_ids=["INATRANDOM-84205978","INATRANDOM-84767787","INATRANDOM-46340202"],from_stage="segment", batch_size=1000,
                      get_color_metrics=True,
                      get_shape_texture_metrics=True,
                      threshold_metrics=[("hls",1,0.20,"below")],
                      pattern_subfolder=None, show=True, print_steps=True, data_folder="D:/bcc/new_random_dragonflies2")

writeBasicColorMetrics(img_ids=["INATRANDOM-6369866","INATRANDOM-84125250","INATRANDOM-116732733"],from_stage="segment", batch_size=1000,
                      get_color_metrics=True,
                      get_shape_texture_metrics=True,
                      threshold_metrics=[("hls",1,0.20,"below")],
                      pattern_subfolder=None, show=True, print_steps=True, data_folder="D:/bcc/new_random_dragonflies2")