import requests
import json
import pandas as pd
from os import listdir
import time
from datetime import datetime
import random
from bigcrittercolor.helpers import _bprint

def downloadiNatRandImgs(n, seed=30, n_before_hr_wait=70, inat_csv_location='', sep=" ", print_steps=True, data_folder=''):
    """
        Download random single images - typically used for creating generalized sets for testing

        :param str genus: the genus name to fix image names for
        :param str proj_root: the path to the project folder
    """

    _bprint(print_steps, "Loading all observation info...")
    data = pd.read_csv(inat_csv_location,sep=sep)

    _bprint(print_steps, "Picking " + str(n) + " random observations...")
    random.seed(seed)
    #ids = random.sample(data, n)['catalogNumber']
    ids = data.sample(n, random_state=seed)['catalogNumber']

    _bprint(print_steps, "Downloading observations...")
    for index, i in enumerate(ids):

        # every 300 observations, sleep for an hour to avoid throttling by iNat server
        if index % n_before_hr_wait == 0 and index != 0:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            _bprint(print_steps, "Downloaded 300 images, waiting one hour until next 300 starting at " + current_time)
            time.sleep(3600)
        try:
            link = 'https://api.inaturalist.org/v1/observations/' + str(i)
            link = link.replace(".0","")
            if not "nan" in link:
                x = requests.get(link)
                obs = json.loads(x.text)

                # parse down to image url
                result = obs.get("results")
                result = result[0]
                result.keys()
                result = result.get('observation_photos')
                result = result[0]
                result = result.get('photo')
                result = result.get('url')

                # replace square with original to get full size
                #result = result.replace("square", "original")
                result = result.replace("square", "medium")
                img = requests.get(result).content
                file = open(data_folder + "/all_images/INATRANDOM-" + str(i) + ".jpg", "wb")

                # write file
                if index%10 == 0: print(str(index))

                file.write(img)
                file.close()
        except ValueError:  # includes simplejson.decoder.JSONDecodeError
            print('Decoding JSON has failed at ' + link + ' probably due to throttling by the iNat server')

    _bprint(print_steps, "Finished downloading random observations")