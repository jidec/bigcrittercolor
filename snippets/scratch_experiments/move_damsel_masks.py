from bigcrittercolor.helpers import _getIDsInFolder
import shutil

ids = _getIDsInFolder("D:/bcc/damsels_segmenter/image")
names = [id + "_mask.png" for id in ids]
sources = ["D:/bcc/unet_training/masks/" + name for name in names]
dests = ["D:/bcc/damsels_segmenter/mask/" + name for name in names]

for src, dst in zip(sources,dests):
    print(src)
    print(dst)
    shutil.copy(src,dst)