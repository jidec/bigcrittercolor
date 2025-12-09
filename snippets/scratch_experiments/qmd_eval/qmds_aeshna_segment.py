from bigcrittercolor import inferMasks

inferMasks(img_ids=None,
  text_prompt="insect", strategy="prompt1",
  aux_segmodel_location="D:/bcc/dfly_training_norm/segmenter.pt", data_folder="E:/aeshna_data")