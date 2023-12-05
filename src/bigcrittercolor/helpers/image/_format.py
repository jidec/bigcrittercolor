import cv2
import numpy as np

def _format(image, in_format, out_format, alpha):
    """ Convert an image from one color format to another and add or removal an alpha channel if specified

        Args:
            image (numpy.ndarray): The input image to be formatted.
            in_format (str): The color format of the input image. Supported formats: 'rgb', 'bgr', 'hls', 'cielab', 'grey', 'grey3', 'binary', 'binary3'.
            out_format (str): The desired color format for the output image. Supported formats: 'rgb', 'bgr', 'hls', 'cielab', 'grey', 'grey3', 'binary', 'binary3'.
            alpha (bool): Specifies whether to add (True) or remove (False) an alpha channel in the output image.

        Returns:
            numpy.ndarray: The reformatted image in the desired color format and with the specified alpha channel handling.
    """
    # Convert the input image to the desired format
    converted_image = _convert_image(image, in_format, out_format)

    # Handle binary conversion
    if 'binary' in out_format:
        _, converted_image = cv2.threshold(converted_image, 127, 255, cv2.THRESH_BINARY)

    # If output is grayscale with 3 channels, convert it to 3 channels
    if out_format == 'grey3':
        converted_image = cv2.cvtColor(converted_image, cv2.COLOR_GRAY2BGR)

    # Handle the alpha channel
    if alpha:
        # Add alpha channel if it doesn't exist
        if len(converted_image.shape) == 2 or converted_image.shape[2] < 4:
            alpha_channel = np.ones(converted_image.shape[:2], dtype=converted_image.dtype) * 255
            converted_image = cv2.merge((converted_image, alpha_channel)) if len(converted_image.shape) == 2 else np.dstack([converted_image, alpha_channel])
    else:
        # Remove alpha channel if it exists
        if len(converted_image.shape) == 3 and converted_image.shape[2] == 4:
            converted_image = converted_image[:, :, :3]

    return converted_image

def _convert_image(image, in_format, out_format):
    image = np.uint8(image)
    if 'grey' in in_format or 'binary' in in_format:
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if 'grey' in out_format or 'binary' in out_format:
        if len(image.shape) == 3:
            converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            converted_image = image
    else:
        conversion_code = _get_conversion_code(in_format, out_format)
        converted_image = cv2.cvtColor(image, conversion_code)

    return converted_image

def _get_conversion_code(in_format, out_format):
    # Define format conversion codes
    format_codes = {
        'rgb': {
            'bgr': cv2.COLOR_RGB2BGR,
            'hls': cv2.COLOR_RGB2HLS,
            'cielab': cv2.COLOR_RGB2Lab
        },
        'bgr': {
            'rgb': cv2.COLOR_BGR2RGB,
            'hls': cv2.COLOR_BGR2HLS,
            'cielab': cv2.COLOR_BGR2Lab
        },
        'hls': {
            'rgb': cv2.COLOR_HLS2RGB,
            'bgr': cv2.COLOR_HLS2BGR,
            #'cielab': cv2.COLOR_HLS2Lab
        },
        'cielab': {
            'rgb': cv2.COLOR_Lab2RGB,
            'bgr': cv2.COLOR_Lab2BGR,
            #'hls': cv2.COLOR_Lab2HLS
        }
    }

    return format_codes[in_format][out_format]

# Example usage
#image = cv2.imread('D:/GitProjects/bigcrittercolor/tests/dummy_bcc_folder/all_images/INAT-215236-1.jpg', cv2.IMREAD_GRAYSCALE) # Load your image
#formatted_image = _format(image, 'grey', 'grey3', False) # Convert from BGR to 3-channel grayscale and add alpha channel

#cv2.imshow('0',formatted_image)
#print(np.shape(formatted_image))
#cv2.waitKey(0)