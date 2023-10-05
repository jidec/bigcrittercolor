import cv2

def _resizeImgToTotalDim(img,total_dim):
    h = img.shape[0]
    w = img.shape[1]
    h_ratio = h / (h + w)
    w_ratio = w / (h + w)

    h_resize = int(h_ratio * total_dim)
    w_resize = int(w_ratio * total_dim)

    img = cv2.resize(img, (w_resize, h_resize), interpolation=cv2.INTER_LINEAR)

    return img

