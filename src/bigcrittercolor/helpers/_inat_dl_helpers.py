import requests
import csv
import datetime
import os.path

base_url = 'https://api.inaturalist.org/v1/'

def getiNatRecords(taxon, research_grade_only=True, lat_lon_box=None, output_file=None, update=True, img_size="medium",
                   data_folder="../.."):
    taxon_id = getTaxonID(taxon)

    base_params = {'taxon_id': taxon_id}
    if research_grade_only:
        base_params['quality_grade'] = 'research'
    #if usa_only:
    #    base_params['swlat'] = 24.396308
    #    base_params['swlng'] = -124.848974
    #    base_params['nelat'] = 49.384358
    #    base_params['nelng'] = -66.885444
    if lat_lon_box is not None:
        base_params['swlat'] = lat_lon_box[0][0]
        base_params['swlng'] = lat_lon_box[0][1]
        base_params['nelat'] = lat_lon_box[1][0]
        base_params['nelng'] = lat_lon_box[1][1]
    ofpath = output_file
    if ofpath is None:
        # ofpath = 'helpers/genus_image_records/iNat_images-' + args.taxon.replace(' ', '_') + '.csv' #EDITED
        # add param for proj_root
        ofpath = data_folder + '/other/inat_download_records/iNat_images-' + taxon.replace(' ',
                                                                                              '_') + '.csv'  # EDITED
    of_exists = os.path.exists(ofpath)
    prev_obs_ids = {}
    if of_exists:
        if update:
            prev_obs_ids = readExtantObsIds(ofpath)
    #prev_obs_ids = readExtantObsIds(ofpath)

    fout = open(ofpath, 'a', encoding='utf-8')
    writer = csv.DictWriter(fout, [
        'obs_id', 'usr_id', 'date', 'latitude', 'longitude', 'taxon',
        'img_cnt', 'img_id', 'file_name', 'img_url', 'width', 'height',
        'license', 'annotations', 'tags', 'download_time', 'family', 'genus', 'species'
    ])
    if not (of_exists):
        writer.writeheader()

    getRecords(base_params, writer, prev_obs_ids, img_size=img_size)


def readExtantObsIds(fpath):
    obs_ids = set()

    with open(fpath, encoding="utf-8") as fin:  # added encoding="utf-8"
        reader = csv.DictReader(fin)
        for row in reader:
            obs_ids.add(int(row['obs_id']))

    return obs_ids

def getRecords(base_params, writer, prev_obs_ids, img_size, vocab=None):
    # ("Records")
    if vocab is None:
        vocab = getControlledVocab()

    params = base_params.copy()
    params['per_page'] = 200

    # Keep track of which observation IDs we've seen to ensure there are no
    # duplicates.
    obs_ids = set()

    # Initialize the stack of time intervals.
    end_ts = time.time()
    start_ts = time.mktime(time.strptime(
        'Jan 01 2008 00:00:00', '%b %d %Y %H:%M:%S'
    ))
    time_stack = [(start_ts, end_ts)]

    TF_STR = '%d %b %Y'

    # Remove time intervals from the stack until the stack is empty, splitting
    # intervals as needed to get the total number of records per interval below
    # 4,000, then downloading all records for each usable interval.
    while len(time_stack) > 0:
        start_ts, end_ts = time_stack.pop()

        s_time_str = datetime.datetime.fromtimestamp(start_ts).isoformat()
        e_time_str = datetime.datetime.fromtimestamp(end_ts).isoformat()
        params['created_d1'] = s_time_str
        params['created_d2'] = e_time_str

        rec_cnt = getRecCnt(params)

        if rec_cnt > 4000:
            print('Splitting interval with {0:,} records ({1} - {2})...'.format(
                rec_cnt, time.strftime(TF_STR, time.localtime(start_ts)),
                time.strftime(TF_STR, time.localtime(end_ts))
            ))
            mid_ts = ((end_ts - start_ts) / 2) + start_ts
            time_stack.append((mid_ts, end_ts))
            time_stack.append((start_ts, mid_ts))
        else:
            print('Getting {0:,} records for interval {1} - {2}...'.format(
                rec_cnt, time.strftime(TF_STR, time.localtime(start_ts)),
                time.strftime(TF_STR, time.localtime(end_ts))
            ))
            retrieveAllRecords(
                params, writer, prev_obs_ids, img_size, obs_ids, vocab
            )

def retrieveAllRecords(
        base_params, writer, prev_obs_ids, img_size, obs_ids, vocab
):
    FNCHARS = string.digits + string.ascii_letters

    params = base_params.copy()
    more_records = True
    record_cnt = 0
    page = 1

    print('Retrieving records...')

    while more_records:
        params['page'] = page
        resp = requests.get(base_url + 'observations', params=params)
        res = resp.json()
        # print(res)
        if len(res['results']) < params['per_page']:
            more_records = False

        row_out = {}
        for rec in res['results']:
            record_cnt += 1

            obs_id = rec['id']
            if obs_id in prev_obs_ids:
                continue

            if obs_id in obs_ids:
                print("Repeat observation encountered")
                # raise Exception(
                #    'Repeat observation encountered: {0}'.format(obs_id)
                # )
            else:
                obs_ids.add(obs_id)

            img_list = rec['photos']

            # if has an image
            if len(img_list) > 0:
                # for each image in img_list
                for i in range(0, len(img_list) - 1):
                    img = img_list[i]
                    img_num = i + 1
                    if (
                            img['url'] is not None and
                            img['original_dimensions'] is not None
                    ):

                        img_fname = '' + str(obs_id) + '-' + str(
                            img_num) + '.jpg'

                        img_url = img['url'].replace('/square.', '/' + img_size + '.')

                        # Get the annotations.
                        annot_list = []
                        for annot in rec['annotations']:
                            attr = annot['controlled_attribute_id']
                            value = annot['controlled_value_id']
                            annot_list.append(
                                '{0}:{1}'.format(vocab[attr], vocab[value])
                            )

                        if rec['location'] is not None:
                            latlong = rec['location'].split(',')
                        else:
                            latlong = ('', '')

                        row_out['obs_id'] = obs_id
                        row_out['usr_id'] = rec['user']['id']
                        row_out['date'] = rec['observed_on']
                        row_out['latitude'] = latlong[0]
                        row_out['longitude'] = latlong[1]
                        row_out['taxon'] = rec['taxon']['name']
                        row_out['img_cnt'] = len(img_list)
                        row_out['img_id'] = img['id']
                        row_out['file_name'] = img_fname
                        row_out['img_url'] = img_url
                        row_out['width'] = img['original_dimensions']['width']
                        row_out['height'] = img['original_dimensions']['height']
                        row_out['license'] = img['license_code']
                        row_out['annotations'] = ','.join(annot_list)
                        row_out['tags'] = ','.join(rec['tags'])
                        row_out['download_time'] = time.strftime(
                            '%Y%b%d-%H:%M:%S', time.localtime()
                        )

                        #print(rec)
                        #print(rec['taxon']['name'])
                        #row_out['superfamily'] = rec['superfamily']
                        #row_out['genus'] = rec['genus']
                        #row_out['species'] = rec['species']

                        writer.writerow(row_out)

        print('  {0:,} records processed...'.format(record_cnt))
        page += 1

# img_records is the PATH to the records .csv
def downloadImages(img_records,imgdir="../../data/other/",no_skip=False,fileout="IMG_RECORDS_FILE-download_log.csv",timeout=20,threads=20):
    downloads = getDownloadRequests(
        img_records, imgdir, not(no_skip)
    )
    if len(downloads) == 0:
        exit()

    # Generate the file name for the failure log.
    logfn = 'fail_log-{0}.csv'.format(
        time.strftime('%Y%b%d-%H:%M:%S', time.localtime())
    )

    # Process the download requests.
    with open(fileout, 'a', encoding="utf-8") as fout: #, open(logfn, 'w') as logf:
        writer = csv.DictWriter(
            fout, ['file_name', 'file_path', 'imgsize', 'bytes', 'img_url', 'time']
        )
        writer.writeheader()
        outrow = {}

        #faillog = csv.DictWriter(
        #    logf, ['file_name', 'img_url', 'time', 'reason']
        #)
        #faillog.writeheader()
        logrow = {}

        for result in mtDownload(
            downloads, imgdir, timeout, threads
        ):
            if result.result == ImageDownloadWorkerResult.SUCCESS:
                outrow['file_name'] = result.identifier
                outrow['file_path'] = result.localpath
                outrow['img_url'] = result.uri
                #outrow['imgsize'] = getImageSize(result.localpath)
                #outrow['bytes'] = os.stat(result.localpath).st_size
                outrow['time'] = result.timestamp
                writer.writerow(outrow)
            elif result.result == ImageDownloadWorkerResult.DOWNLOAD_FAIL:
                logrow['file_name'] = result.identifier
                logrow['img_url'] = result.uri
                logrow['time'] = result.timestamp
                logrow['reason'] = result.fail_reason
                #faillog.writerow(logrow)

# Downloads images from a set of iDigBio search results, using random names for
# the downloaded files, and generates a CSV file mapping downloaded file names
# to source URIs, iDigBio core IDs, and scientific names.

def getDownloadRequests(fpath, img_dir, skip_existing):
    """
    Generates a list of download request (file_name, URI) tuples.
    """
    downloads = []
    with open(fpath, encoding="utf-8") as fin:
        reader = csv.DictReader(fin)

        for line in reader:
            imguri = line['img_url']
            fname = line['file_name']
            tmp_fname = os.path.splitext(fname)[0]
            downloads.append((tmp_fname, imguri))

    return downloads
# ("Starting")

def getTaxonID(taxon_name):
    if len(taxon_name.split(' ')) == 2:
        rank = 'species'
    else:
        rank = 'genus'

    params = {'rank': rank, 'q': taxon_name}
    resp = requests.get(base_url + 'taxa', params=params)
    res = resp.json()

    match_cnt = 0
    taxon_id = None
    taxon_ids = []
    for t_info in res['results']:
        if t_info['name'] == taxon_name:
            taxon_id = t_info['id']
            taxon_ids.append(t_info['id'])
            match_cnt += 1

    if match_cnt == 0:
        raise Exception('Could not find the {0} "{1}".\n'.format(
            rank, taxon_name
        ))
    elif match_cnt > 1:
        # raise Exception(
        # 'More than one {0} name match for "{1}".\n'.format(rank, taxon_name
        # ))
        print("Genus matches " + str(match_cnt) + " using the first")
        print(len(taxon_ids))
        taxon_id = taxon_ids[0]
        print(taxon_ids)
        print(taxon_id)

    return taxon_id


def getRecCnt(base_params):
    # print("Getting record count")
    resp = requests.get(base_url + 'observations', params=base_params)
    res = resp.json()

    return int(res['total_results'])


def getControlledVocab():
    # print("Getting controlled vocab")
    """
    Retrieves the controlled vocabulary used for annotations.
    """
    resp = requests.get(base_url + 'controlled_terms')
    res = resp.json()

    vocab = {}

    for result in res['results']:
        vocab[result['id']] = result['label']
        for val in result['values']:
            vocab[val['id']] = val['label']

    return vocab

#
# Utility classes and functions for working with image files at the file system
# level, including facilities for parallel processing.
#

import os
import os.path
import subprocess
import time
import imghdr
from PIL import Image
import multiprocessing as mp
import threading as mt
from queue import Queue as MTQueue
import string
import random
import urllib.request as req
import shutil

IMG_EXTS = (
    '.jpg', '.jpeg', '.JPG', '.JPEG', '.tif', '.tiff', '.TIF', '.TIFF',
    '.png', '.PNG'
)

def findImageFiles(dirname, sort=True, cs_sort=False, inspect=False):
    """
    Returns a list of all raster image files in a specified directory.  The
    list elements are full paths (either relative or absolute, depending on the
    value of "dirname").  If "inspect" is True, the actual contents of each
    file will be inspected to determine if the file is an image file.  If
    False, file name extensions will be used to determine file type.  If
    possible, "inspect" should be False for processing large image libraries
    because it is much faster.  On a typical laptop running GNU/Linux with
    Python 2.7, file contents inspection takes ~2.37 times longer than file
    name extension filtering.  Oddly, the performance gap is even worse with
    Python 3.5, where file contents inspection takes ~2.89 times longer.

    sort: If True, the file names are returned in ascending alphanumeric sorted
        order.
    cs_sort: If True, the sort on file names will be case-sensitive.
    inspect: If True, inspect file contents to determine file types; otherwise,
        use file name extensions to determine file types.
    """
    fnames = []

    for fname in os.listdir(dirname):
        fpath = os.path.join(dirname, fname)
        if os.path.isfile(fpath):
            if inspect:
                if imghdr.what(fpath) is not None:
                    fnames.append(fpath)
            elif os.path.splitext(fpath)[1] in IMG_EXTS:
                fnames.append(fpath)

    if sort:
        if cs_sort:
            fnames.sort()
        else:
            fnames.sort(key=str.lower)

    return fnames

def getImageSize(fpath):
    """
    Returns the size of an image file in pixels.  The size is returned as the
    tuple (width, height).
    """
    img = Image.open(fpath)
    size = img.size
    img.close()
    
    return size

def convertToJPEG(fpath):
    """
    Takes a raw image file and converts it to JPEG format, if needed, and adds
    the ".jpg" extension to the file name, if needed.  Returns the new file
    name.  To support JPEG 2000, this function currently uses a call to the
    external "file" utility for file type detection and uses GraphicsMagick for
    image format conversion.
    """
    # Second command is for old version of the file utility.
    #cmdargs = ['file', '-b', '--parameter', 'name=300', fpath]
    cmdargs = ['file', '-b', fpath]

    result = subprocess.run(
        cmdargs, check=True, stdout=subprocess.PIPE, universal_newlines=True
    )
    resultstr = result.stdout

    newname = os.path.splitext(fpath)[0] + '.jpg'

    if resultstr.startswith('JPEG image data'):
        os.rename(fpath, newname)
    elif resultstr.split()[0] in ('JPEG', 'PNG', 'TIFF', 'GIF', 'Minix'):
        # There is a strange bug in older versions of file that causes some
        # JPEG files to be reported as "Minix filesystem, V2".  If we encounter
        # that, verify that it is a JPEG file using the "identify" utility.
        if resultstr.startswith('Minix'):
            cmdargs = ['identify', fpath]
            result = subprocess.run(
                cmdargs, check=True, stdout=subprocess.PIPE,
                universal_newlines=True
            )
            if 'JPEG' not in result.stdout:
                raise Exception(
                    'Unsupported image format encountered in file {0}: '
                    '"{1}".'.format(fpath, result)
                )

        # First command is for GraphicsMagick; second is for ImageMagick.
        #cmdstr = 'gm convert -quality 97% {0} {1}'.format(fpath, newname)
        cmdstr = 'convert -quality 97% {0} {1}'.format(fpath, newname)
        subprocess.run(cmdstr, check=True, shell=True)
        os.unlink(fpath)
    else:
        raise Exception(
            'Unsupported image format encountered in file {0}: "{1}".'.format(
                fpath, result
            )
        )

    return newname


class MTListReader:
    """
    Implements a thread-safe wrapper for sequentially accessing items in a list
    or other integer-indexible object that supports len().  Includes an error
    flag, self.error, to indicate whether processing should continue.  If the
    flag is set, MTListReader.STOPVAL will be returned by all subsequent calls
    to nextItem().
    """
    STOPVAL = None

    def __init__(self, data):
        """
        data: An integer-indexible object that supports len().
        """
        self.data = data
        self.index = 0
        self.lock = mt.Lock()
        self.error = mt.Event()

    def __len__(self):
        return len(self.data)

    def nextItem(self):
        """
        Returns the next item in the sequence along with its index as the tuple
        (index, item).  At the end of the sequence, or if the error event is
        set, returns None.
        """
        with self.lock:
            retval = self.STOPVAL
            if self.index < len(self.data) and not(self.error.is_set()):
                retval = (self.index, self.data[self.index])
                self.index += 1

        return retval


class ImageDownloadWorkerResult:
    """
    This is a simple struct-like class that ImageDownloadWorker instances use
    to report the results of download tasks.
    """
    SUCCESS = 0
    DOWNLOAD_FAIL = 1
    ERROR = 2

    def __init__(self):
        # Indicates the overall result of the download operation.
        self.result = None
        # An (optional) identifier for the download operation, separate from
        # the URI.
        self.identifier = None
        # The remote resource URI.
        self.uri = None
        # The path for the local copy of the resource, if the download was
        # successful.
        self.localpath = None
        # The time at which the download was completed.
        self.timestamp = None
        # The reason for a download failure, if known.
        self.fail_reason = None
        # Any exception generated by the download operation.
        self.exception = None


class ImageDownloadWorker(mt.Thread):
    """
    A subclass of multithreading.Thread designed for downloading image files.

    I investigated three alternative implementations for HTTP downloads:
    requests, urllib, and wget (using subprocesses).  I tested the performance
    of all three using a sample of 100 image URIs from iDigBio.  For the
    requests and urllib implementations, I also experimented with different
    chunk/buffer sizes.  For all implementations, I experimented with different
    numbers of threads: 1, 10, 20, and 40.  I found that urllib consistently
    produced the best results, with the requests and wget implementations
    performing similarly, but neither as fast as urllib.  Larger chunk sizes (1
    KB or greater) significantly improved the performance of requests, but even
    going up to 1 MB chunks, it still was slower than urllib.  Although the
    documentation suggests that None should work as a chunk size, the downloads
    always hung when I tried this.  The default urllib buffer size (16*1024)
    seemed to work quite well, but I found some evidence that performance
    improved very slightly when I increased this to 1 MB, so that is the
    default.  On the test set, 10, 20, and 40 threads all performed similarly,
    but 20 appeared to be the optimal number for the test set, so that is the
    default.  Here are sample benchmarks from server-class hardware for urllib
    with a 1 MB buffer size (best time of 3 trials for each test):
         1 thread : 2 m, 19.16 s
        10 threads: 0 m, 20.10 s
        20 threads: 0 m, 18.82 s
        40 threads: 0 m, 20.37 s
    """
    STOPVAL = MTListReader.STOPVAL

    def __init__(
        self, downloads, outputq, outputdir, timeout=20, update_console=True
    ):
        """
        downloads: An MTListReader for retrieving download requests.
        outputq: A Queue for reporting back image download results.
        outputdir: A directory in which to save downloaded images.
        timeout: Seconds to wait before connect or read timeouts.
        update_console: If True, send download updates to the console.
        """
        super(ImageDownloadWorker, self).__init__()

        self.downloads = downloads
        self.outputq = outputq
        self.outputdir = outputdir
        self.timeout = timeout
        self.update_console = update_console

    def _isImageIncomplete(self, fpath):
        """
        Tries to determine if an image download completely failed or was only
        partially successful.  Returns True if the image download failed, False
        otherwise.  Because the default Pillow binary does not include JPEG
        2000 support, partial downloads are not detected for JPEG 2000 files.
        """
        if not(os.path.exists(fpath)):
            return True

        if os.stat(fpath).st_size == 0:
            return True

        try:
            img = Image.open(fpath)
        except:
            return True

        if img.format != 'JPEG2000':
            try:
                pxdata = img.load()
            except IOError:
                return True

        return False

    def run(self):
        for index, item in iter(self.downloads.nextItem, self.STOPVAL):
            fname, uri = item

            result = ImageDownloadWorkerResult()
            result.uri = uri
            result.identifier = fname
            if self.update_console:
                print(
                    '({1:.1f}%) Downloading {0}...'.format(
                        uri, ((index + 1) / len(self.downloads)) * 100
                    )
                )

            try:
                imgfpath = os.path.join(self.outputdir, fname)
                reqerr = None

                try:
                    httpr = req.urlopen(uri, timeout=self.timeout)
                    with open(imgfpath, 'wb') as imgout:
                        shutil.copyfileobj(httpr, imgout, 1024*1024)
                except OSError as err:
                    # Note that urllib.error.URLError and all low-level socket
                    # exceptions are subclasses of OSError.
                    reqerr = err

                result.timestamp = time.strftime(
                    '%Y%b%d-%H:%M:%S', time.localtime()
                )

                # Check if the download succeeded.
                if reqerr is None and not(self._isImageIncomplete(imgfpath)):
                    #newfpath = convertToJPEG(imgfpath) #CHANGED
                    result.result = ImageDownloadWorkerResult.SUCCESS
                    #result.localpath = newfpath
                else:
                    result.result = ImageDownloadWorkerResult.DOWNLOAD_FAIL
                    if reqerr is not None:
                        result.fail_reason = str(reqerr)

                    if os.path.exists(imgfpath):
                        os.unlink(imgfpath)

                self.outputq.put(result)
            except Exception as err:
                result.result = ImageDownloadWorkerResult.ERROR
                result.exception = err
                self.outputq.put(result)


def mtDownload(downloads, imgdir, timeout=20, maxthread_cnt=20):
    """
    Initiates multithreaded downloading of a list of download requests.  This
    function returns a generator object that can be used to iterate over all
    download request results.

    downloads: A list (or other integer-indexible sequence that supports len())
        of (imageURI, identifer) pairs.
    imgdir: Directory in which to save downloaded images.
    timeout: Seconds to wait before connect or read timeouts.
    maxthread_cnt: The maximum number of threads to use.
    """
    downloadscnt = len(downloads)
    mtdownloads = MTListReader(downloads)

    if downloadscnt >= maxthread_cnt:
        tcnt = maxthread_cnt
    else:
        tcnt = downloadscnt

    outputq = MTQueue()

    threads = []
    for cnt in range(tcnt):
        thread = ImageDownloadWorker(mtdownloads, outputq, imgdir, timeout)
        thread.daemon = True
        threads.append(thread)
        thread.start()

    lasterror = None

    resultcnt = 0
    while resultcnt < downloadscnt and not(mtdownloads.error.is_set()):
        result = outputq.get()
        resultcnt += 1
        if result.result == ImageDownloadWorkerResult.ERROR:
            mtdownloads.error.set()
            lasterror = result
        else:        
            yield result

    for thread in threads:
        thread.join()

    if lasterror is not None:
        raise lasterror.exception

