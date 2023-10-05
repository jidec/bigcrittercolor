from bigcrittercolor.clustextract import createInitialFilter

def test_createfilter():
    createInitialFilter(test_proportion=0.2, equalize_good_bad=True, data_folder="E:/aeshna_data")
test_createfilter()