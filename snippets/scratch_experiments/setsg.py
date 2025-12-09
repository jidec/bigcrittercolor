from bigcrittercolor import filterExtractSegs
from bigcrittercolor import writeColorMetrics

writeColorMetrics(from_stage="segment",batch_size=50,data_folder="D:/bcc/ringtails")

#filterExtractSegs(used_aux_segmodel=True,batch_size=50,data_folder="D:/bcc/ringtails")