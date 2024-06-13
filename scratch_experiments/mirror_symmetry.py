import sys
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob

# create SIFT object ((a feature detection algorithm))
sift = cv2.xfeatures2d.SIFT_create()
# create BFMatcher object
bf = cv2.BFMatcher()


####################
## Run whole process
####################

def detecting_mirrorLine(image, title, show_detail=False):
    """
    Main function

    If show_detail = True, plot matching details
    """
    # create mirror object
    mirror = Mirror_Symmetry_detection(image)

    # extracting and Matching a pair of symmetric features
    matchpoints = mirror.find_matchpoints()

    # get r, tehta (polar coordinates) of the midpoints of all pair of symmetric features
    points_r, points_theta = mirror.find_points_r_theta(matchpoints)

    if show_detail:  # visualize process in detail
        mirror.draw_matches(matchpoints, top=10)
        mirror.draw_hex(points_r, points_theta)

    # find the best one with highest vote
    image_hexbin = plt.hexbin(points_r, points_theta, bins=200, cmap=plt.cm.Spectral_r)
    sorted_vote = mirror.sort_hexbin_by_votes(image_hexbin)
    r, theta = mirror.find_coordinate_maxhexbin(image_hexbin, sorted_vote, vertical=False)

    # add mirror line based on r and theta
    mirror.draw_mirrorLine(r, theta, title)


def test_case(filesPath):
    files = sorted([f for f in glob.glob(filesPath)])
    for file in files:
        detecting_mirrorLine(file, "With Mirror Line")


#############################
## Mirror symmetry detection
#############################


class Mirror_Symmetry_detection:
    def __init__(self, image):
        self.image = image  # convert the image into the array/matrix

        self.reflected_image = np.fliplr(self.image)  # Flipped version of image

        # find the keypoints and descriptors with SIFT
        self.kp1, self.des1 = sift.detectAndCompute(self.image, None)
        self.kp2, self.des2 = sift.detectAndCompute(self.reflected_image, None)

    def find_matchpoints(self):
        """
        Extracting and Matching a pair of symmetric features

        Matches are then sort between the features ki and the mirrored features mj
        to form a set of (pi,pj) pairs of potentially symmetric features.

        Ideally a keypoint at a certain spot on the object in original image should have a descriptor very similar to
        the descriptor on a point on the object in its mirrored version
        """
        # use BFMatcher.knnMatch() to get （k=2）matches
        matches = bf.knnMatch(self.des1, self.des2, k=2)
        # these matches are equivalent only one need be recorded
        matchpoints = [item[0] for item in matches]

        # sort to determine the dominant symmetries
        # Distance between descriptors. The lower, the better it is.
        matchpoints = sorted(matchpoints, key=lambda x: x.distance)

        return matchpoints

    def find_points_r_theta(self, matchpoints: list):
        """
        Get r, tehta of the midpoints of all pair of symmetric features
        """
        points_r = []  # list of r for each point
        points_theta = []  # list of theta for each point
        for match in matchpoints:

            point = self.kp1[match.queryIdx]  # queryIdx is an index into one set of keypoints, (origin image)
            mirpoint = self.kp2[match.trainIdx]  # trainIdx is an index into the other set of keypoints (fliped image)

            mirpoint.angle = np.deg2rad(mirpoint.angle)  # Normalise orientation
            mirpoint.angle = np.pi - mirpoint.angle
            # convert angles to positive
            if mirpoint.angle < 0.0:
                mirpoint.angle += 2 * np.pi

            # pt: coordinates of the keypoints x:pt[0], y:pt[1]
            # change x, not y
            mirpoint.pt = (self.reflected_image.shape[1] - mirpoint.pt[0], mirpoint.pt[1])

            # get θij: the angle this line subtends with the x-axis.
            theta = angle_with_x_axis(point.pt, mirpoint.pt)

            # midpoit (xc,yc) are the image centred co-ordinates of the mid-point of the line joining pi and pj
            xc, yc = midpoint(point.pt, mirpoint.pt)
            r = xc * np.cos(theta) + yc * np.sin(theta)

            points_r.append(r)
            points_theta.append(theta)

        return points_r, points_theta  # polar coordinates

    def draw_matches(self, matchpoints, top=10):
        """visualize the best matchs
        """
        img = cv2.drawMatches(self.image, self.kp1, self.reflected_image, self.kp2,
                              matchpoints[:top], None, flags=2)
        plt.imshow(img);
        plt.title("Top {} pairs of symmetry points".format(top))
        plt.show()

    def draw_hex(self, points_r: list, points_theta: list):
        """
        Visualize hex bins based on r and theta
        """
        # Make a 2D hexagonal binning plot of points r and theta
        image_hexbin = plt.hexbin(points_r, points_theta, bins=200, cmap=plt.cm.Spectral_r)
        plt.colorbar()  # add color bar
        plt.show()

    def find_coordinate_maxhexbin(self, image_hexbin, sorted_vote, vertical):
        """Try to find the x and y coordinates of the hexbin with max count
        """
        for k, v in sorted_vote.items():
            # if mirror line is vertical, return the highest vote
            if vertical:
                return k[0], k[1]
            # otherwise, return the highest vote, whose y is not 0 or pi
            else:
                if k[1] == 0 or k[1] == np.pi:
                    continue
                else:
                    return k[0], k[1]

    def sort_hexbin_by_votes(self, image_hexbin):
        """Sort hexbins by decreasing count. (lower vote)
        """
        counts = image_hexbin.get_array()
        ncnts = np.count_nonzero(np.power(10, counts))  # get non-zero hexbins
        verts = image_hexbin.get_offsets()  # coordinates of each hexbin
        output = {}

        for offc in range(verts.shape[0]):
            binx, biny = verts[offc][0], verts[offc][1]
            if counts[offc]:
                output[(binx, biny)] = counts[offc]
        return {k: v for k, v in sorted(output.items(), key=lambda item: item[1], reverse=True)}

    def draw_mirrorLine(self, r, theta, title: str):
        """
        Draw mirror line based on r theta polar co-ordinate
        """
        for y in range(len(self.image)):
            try:
                x = int((r - y * np.sin(theta)) / np.cos(theta))
                self.image[y][x] = 255
                self.image[y][x + 1] = 255
            except IndexError:
                continue

        # draw plot
        plt.imshow(self.image)
        plt.axis('off')
        plt.title(title)
        plt.show()


def angle_with_x_axis(pi, pj):  # 公式在文件里解释
    """
    calculate θij:
        the angle this line subtends with the x-axis.
    """
    # get the difference between point p1 and p2
    x, y = pi[0] - pj[0], pi[1] - pj[1]

    if x == 0:
        return np.pi / 2

    angle = np.arctan(y / x)
    if angle < 0:
        angle += np.pi
    return angle


def midpoint(pi, pj):
    """
    get x and y coordinates of the midpoint of pi and pj
    """
    return (pi[0] + pj[0]) / 2, (pi[1] + pj[1]) / 2

path = "D:/dfly_appr_expr/appr1/masks/INATRANDOM-6041658_mask.png"
#path = "D:/GitProjects/bigcrittercolor/tests/temp/butterfly.png"

from bigcrittercolor.helpers import _readBCCImgs

img = _readBCCImgs(img_ids=["INAT-169503469-2"], type="mask",data_folder="E:/aeshna_data_appr1")

from skimage.morphology import skeletonize
Wimg = skeletonize(img[0])
detecting_mirrorLine(img, "butterflywith Mirror Line", show_detail = True)

import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

sift = cv2.xfeatures2d.SIFT_create()

def very_close(a, b, tol=4.0):
    """Checks if the points a, b are within
    tol distance of each other."""
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) < tol


def S(si, sj, sigma=1):
    """Computes the 'S' function mentioned in
    the research paper."""
    q = (-abs(si - sj)) / (sigma * (si + sj))
    return np.exp(q ** 2)


def reisfeld(phi, phj, theta):
    return 1 - np.cos(phi + phj - 2 * theta)


def midpoint(i, j):
    return (i[0] + j[0]) / 2, (i[1] + j[1]) / 2


def angle_with_x_axis(i, j):
    x, y = i[0] - j[0], i[1] - j[1]
    if x == 0:
        return np.pi / 2
    angle = np.arctan(y / x)
    if angle < 0:
        angle += np.pi
    return angle


def superm2(image):
    """Performs the symmetry detection on image.
    Somewhat clunky at the moment -- first you
    must comment out the last two lines: the
    call to `draw` and `cv2.imshow` and uncomment
    `hex` call. This will show a 3d histogram, where
    bright orange/red is the maximum (most voted for
    line of symmetry). Manually get the coordinates,
    and re-run but this time uncomment draw/imshow."""
    mimage = np.fliplr(image)
    kp1, des1 = sift.detectAndCompute(image, None)
    kp2, des2 = sift.detectAndCompute(mimage, None)
    for p, mp in zip(kp1, kp2):
        p.angle = np.deg2rad(p.angle)
        mp.angle = np.deg2rad(mp.angle)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    houghr = np.zeros(len(matches))
    houghth = np.zeros(len(matches))
    weights = np.zeros(len(matches))
    i = 0
    good = []
    for match, match2 in matches:
        point = kp1[match.queryIdx]
        mirpoint = kp2[match.trainIdx]
        mirpoint2 = kp2[match2.trainIdx]
        mirpoint2.angle = np.pi - mirpoint2.angle
        mirpoint.angle = np.pi - mirpoint.angle
        if mirpoint.angle < 0.0:
            mirpoint.angle += 2 * np.pi
        if mirpoint2.angle < 0.0:
            mirpoint2.angle += 2 * np.pi
        mirpoint.pt = (mimage.shape[1] - mirpoint.pt[0], mirpoint.pt[1])
        if very_close(point.pt, mirpoint.pt):
            mirpoint = mirpoint2
            good.append(match2)
        else:
            good.append(match)
        theta = angle_with_x_axis(point.pt, mirpoint.pt)
        xc, yc = midpoint(point.pt, mirpoint.pt)
        r = xc * np.cos(theta) + yc * np.sin(theta)
        Mij = reisfeld(point.angle, mirpoint.angle, theta) * S(
            point.size, mirpoint.size
        )
        houghr[i] = r
        houghth[i] = theta
        weights[i] = Mij
        i += 1
    # matches = sorted(matches, key = lambda x:x.distance)
    good = sorted(good, key=lambda x: x.distance)

    def draw(r, theta):
        if np.pi / 4 < theta < 3 * (np.pi / 4):
            for x in range(len(image.T)):
                y = int((r - x * np.cos(theta)) / np.sin(theta))
                if 0 <= y < len(image.T[x]):
                    image[y][x] = 255
        else:
            for y in range(len(image)):
                x = int((r - y * np.sin(theta)) / np.cos(theta))
                if 0 <= x < len(image[y]):
                    image[y][x] = 255

    img3 = cv2.drawMatches(image, kp1, mimage, kp2, good[:15], None, flags=2)
    # print(*(m.distance for m in matches[:10]))
    # cv2.imshow('a',img3); cv2.waitKey(0);
    def hex():
        plt.hexbin(houghr, houghth, bins=200)
        plt.show()

    hex()
    # draw(2.8, 2.4)
    # cv2.imshow('a', image); cv2.waitKey(0);


def draw(image, r, theta):
    if np.pi / 4 < theta < 3 * (np.pi / 4):
        for x in range(len(image.T)):
            y = int((r - x * np.cos(theta)) / np.sin(theta))
            if 0 <= y < len(image.T[x]):
                image[y][x] = 255
    else:
        for y in range(len(image)):
            x = int((r - y * np.sin(theta)) / np.cos(theta))
            if 0 <= x < len(image[y]):
                image[y][x] = 255


def main():
    argc = len(sys.argv)
    if not (argc == 2 or argc == 4 or argc == 5):
        print("Usage: python3 detect.py IMAGE [r] [theta]")
        return
    if argc == 2:
        superm2(cv2.imread(sys.argv[1], 0))
    elif argc == 4:
        image = cv2.imread(sys.argv[1], 0)
        draw(image, float(sys.argv[2]), float(sys.argv[3]))
        cv2.imshow("a", image)
        cv2.waitKey(0)
    else:
        image = cv2.imread(sys.argv[1], 0)
        draw(image, float(sys.argv[2]), float(sys.argv[3]))
        cv2.imwrite("{}".format(sys.argv[4]), image)


if __name__ == "__main__":
    main()