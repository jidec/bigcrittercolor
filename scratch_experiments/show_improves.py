from bigcrittercolor.helpers import _showImages
from bigcrittercolor.project import showBCCImages

showBCCImages(type="image",data_folder="D:/bcc/ringtails",num_cols=4,sample_n=16)
showBCCImages(type="mask",data_folder="D:/bcc/ringtails",num_cols=12,sample_n=36)
showBCCImages(type="segment",data_folder="D:/bcc/ringtails",num_cols=12,sample_n=36)
