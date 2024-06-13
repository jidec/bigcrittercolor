import cv2
import numpy as np
import os
from bigcrittercolor.helpers import _getIDsInFolder

def samTemplateMatchHack(folder, reverse=False):
    filenames = _getIDsInFolder(folder + "/source")
    filenames = [name + "_segment" for name in filenames]
    for name in filenames:
        print(name)
        source = cv2.imread(folder + "/source/" + name + ".png")
        template = cv2.imread(folder + "/template/" + name + "_t.png")
        # Use the template matching function
        result = cv2.matchTemplate(source, template, cv2.TM_CCOEFF_NORMED)

        # Find the location with the highest correlation
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # Define the top-left corner of the rectangle (starting point)
        start_point = max_loc
        topleft_x = start_point[0]
        topleft_y = start_point[1]

        # Define the bottom-right corner of the rectangle
        end_point = (start_point[0] + template.shape[1], start_point[1] + template.shape[0])
        botright_x = end_point[0]
        botright_y = end_point[1]

        # Draw a rectangle around the matched region
        #cv2.rectangle(source, start_point, end_point, color=(255), thickness=2)

        # Display the result
        #cv2.imshow("Matched Region", source)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        empty = np.zeros_like(source)

        # Place the overlay image on the base image
        empty[topleft_y:botright_y, topleft_x:botright_x] = template

        _, thresholded = cv2.threshold(empty, 0, 255, cv2.THRESH_BINARY)

        if reverse:
            subtr = cv2.subtract(source, thresholded)
            _, thresholded = cv2.threshold(subtr, 0, 255, cv2.THRESH_BINARY)
            thresholded = cv2.cvtColor(thresholded,cv2.COLOR_BGR2GRAY)
            # Find the contours
            contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Sort the contours by area and pick the largest
            largest_contour = max(contours, key=cv2.contourArea)
            empty = np.zeros_like(source)
            # Fill the largest contour with white
            cv2.drawContours(empty, [largest_contour], -1, (255,255,255), thickness=cv2.FILLED)
            thresholded = empty
            thresholded = cv2.erode(thresholded, kernel=np.ones(3, np.uint8),iterations=1)
            #cv2.imshow("0",thresholded)
            #cv2.waitKey(0)

        #cv2.imshow("new", thresholded)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        cv2.imwrite(folder + "/out/" + name + "_mask.png",thresholded)


samTemplateMatchHack("D:/GitProjects/bigcrittercolor/tests/sam_hack",reverse=True)