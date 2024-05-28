from bigcrittercolor import inferMasks

ids = ["INAT-1828591-1", "INAT-7321648-1"]
ids = ["INAT-163633-1"]
inferMasks(img_ids=ids,skip_existing=False, aux_segmodel_location="D:/bcc/dfly_training_norm/segmenter.pt",
  text_prompt="insect", strategy="prompt1",
  show_indv=True, print_steps=True, print_details=True,
  data_folder="D:/bcc/download_test")