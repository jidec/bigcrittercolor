import cv2
import numpy as np

# finds the longest line that can be drawn through the sole white blob in a binary image by:
# 1. getting the minimum area rectangle
# 2. finding all points in the top face, AND the bot face
# 3. drawing lines between every top face point and every bottom face point
# 4. finding the line containing the most white pixels (the longest line)

# returns a tuple of points for the line
# GREYSCALE uint8 image is taken
def _getLongestLineThruBlob(binary_img, line_point_interval=15, dilate_kernel=10, show=False):

    # apply dilation
    if dilate_kernel is not None:
        binary_img = cv2.dilate(binary_img, np.ones((dilate_kernel,dilate_kernel), np.uint8), iterations=1)

    # find contours in the binary image
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        print("No contour found, skipping getting longest line...")
    else:
        # find the largest contour (the white blob)
        largest_contour = max(contours, key=cv2.contourArea)

        # fit a minimum area rectangle around the contour
        min_area_rect = cv2.minAreaRect(largest_contour)

        # draw the minimum area rectangle on the original image (optional)
        box = cv2.boxPoints(min_area_rect).astype(int)

        # display the image with the minimum area rectangle (optional)
        if show:
            img_with_box = np.copy(binary_img)
            cv2.drawContours(img_with_box, [box], 0, 255, 2)
            cv2.imshow('Image with Minimum Area Rectangle', img_with_box)
            cv2.waitKey(0)

        rect_points = cv2.boxPoints(min_area_rect)

        # Sort the points by y-coordinate in ascending order
        rect_points = sorted(rect_points, key=lambda x: x[1])

        # The top face of the rectangle is the one with the smallest y-coordinate
        top_face_points = rect_points[:2]  # Assuming the rectangle has four corners
        bot_face_points = rect_points[2:4]  # Assuming the rectangle has four corners

        xsorted_points = sorted(cv2.boxPoints(min_area_rect), key=lambda point: point[0])
        left_face_points = xsorted_points[:2]  # Assuming the rectangle has four corners
        right_face_points = xsorted_points[2:4]  # Assuming the rectangle has four corners

        # Draw the top face on a copy of the original image
        result_image = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(result_image, [np.int0(top_face_points)], 0, (0, 255, 0), 2)
        cv2.drawContours(result_image, [np.int0(left_face_points)], 0, (0, 255, 0), 2)

        # Display the result
        if show:
            cv2.imshow('Top and Left Face of Rectangle', result_image)
            cv2.waitKey(0)

        def _points_on_line(start, end):
            idx = np.round(np.array(start)).astype(int)
            end_idx = np.round(np.array(end)).astype(int)
            points = [idx]

            if np.all(idx == end_idx):
                return points

            diff = np.array(end, dtype=float) - np.array(start, dtype=float)
            direction = (diff / np.abs(diff)).astype(int)
            coord = np.array(start, dtype=float)

            while np.any(idx != end_idx):
                # compute how far we need to go to reach the side of the pixel at idx
                t = (idx + direction / 2 - coord) / diff
                i = np.argmin(t)
                coord += t[i] * diff
                idx = idx.copy()
                idx[i] += direction[i]
                points.append(idx)

            return points

        top_line_coordinates = _points_on_line(top_face_points[0], top_face_points[1])
        top_line_coordinates = top_line_coordinates[0::line_point_interval]

        bot_line_coordinates = _points_on_line(bot_face_points[0], bot_face_points[1])
        bot_line_coordinates = bot_line_coordinates[0::line_point_interval]

        # get left as well
        left_line_coordinates = _points_on_line(left_face_points[0], left_face_points[1])
        left_line_coordinates = left_line_coordinates[0::line_point_interval]
        right_line_coordinates = _points_on_line(right_face_points[0], right_face_points[1])
        right_line_coordinates = right_line_coordinates[0::line_point_interval]
        # then append
        top_line_coordinates = top_line_coordinates + left_line_coordinates
        bot_line_coordinates = bot_line_coordinates + right_line_coordinates

        # count the number of white pixels in a line from the start to end
        def _count_white_pixels_along_line(image, start_coords, end_coords):
            # create a copy of the input image to avoid modifying the original image
            image_copy = np.copy(image)

            # define line coordinates
            x1, y1 = start_coords
            x2, y2 = end_coords

            # Bresenham's line algorithm to get the coordinates of the line pixels
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            sx = 1 if x1 < x2 else -1
            sy = 1 if y1 < y2 else -1
            err = dx - dy

            white_pixel_count = 0
            entered_white = False
            exited_white = False
            while True:
                # check if the current coordinates are within the image boundaries
                if x1 >= 0 and y1 >= 0 and x1 < image.shape[1] and y1 < image.shape[0]:
                    # get the pixel value at the current coordinates
                    pixel_value = image_copy[y1, x1]

                    # assuming the image is in grayscale, check if the pixel is white (255)
                    if pixel_value == 255:
                        white_pixel_count += 1
                        if not entered_white:
                            entered_white = True
                        # if we exited white already, then we are REENTERING white and therefore this is a bad line so set to 0
                        if exited_white:
                            white_pixel_count = 0
                    # if pixel is black AND we already entered white
                    elif entered_white:
                        # we have exited white
                        exited_white = True

                if x1 == x2 and y1 == y2:
                    break

                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x1 += sx
                if e2 < dx:
                    err += dx
                    y1 += sy

            return white_pixel_count

        tops = []
        bots = []
        nwhite = []

        for t in top_line_coordinates:
            for b in bot_line_coordinates:
                n = _count_white_pixels_along_line(binary_img, start_coords=tuple(t), end_coords=tuple(b))
                tops.append(t)
                bots.append(b)
                nwhite.append(n)

        maxindex = nwhite.index(max(nwhite))
        maxtop = tops[maxindex]
        maxbot = bots[maxindex]

        if show:
            line_img = np.copy(binary_img)
            cv2.line(line_img, maxtop, maxbot, (0, 255, 255), 2)
            cv2.imshow("Image with found line", line_img)
            cv2.waitKey(0)

        return ((maxtop, maxbot))