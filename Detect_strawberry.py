import cv2
from matplotlib import pyplot as plt
import numpy as np 
from math import sin, cos

green = (0, 255, 0)

def show(image):
    # fig size in inches
    plt.figure(figsize=(10, 10))
    plt.imshow(image, interpolation='nearest')


def overlay_mask(mask, image):
    #make mask RGB
    rgb_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    img = cv2.addWeighted(rgb_mask, 0.5, image, 0.5, 0)
    return img


def find_biggest_contour(image):
    
    # Copy to prevent modification
    image = image.copy()
    _, contours, hierarchy= cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Isolate largest contour
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
 
    mask = np.zeros(image.shape, np.uint8)
    cv2.drawContours(mask, [biggest_contour], -1, 255, -1)
    return biggest_contour, mask

def circle_contour(image, contour):
    # bounding ellipse
    image_with_ellipse = image.copy()
    ellipse = cv2.fitEllipse(contour)

    # add it
    cv2.ellipse(image_with_ellipse, ellipse, green, 2, cv2.LINE_AA)
    return image_with_ellipse

def findStrawberry(image):
    # covert image to color scheme we want
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #Scaling our image
    dim_limit = max(image.shape)
    scale = 700/dim_limit
    image = cv2.resize(image, None, fx=scale, fy=scale)

    # Cleaning our image [making it smooth]
    image_blurred = cv2.GaussianBlur(image, (7, 7), 0)
    image_blurred_hsv = cv2.cvtColor(image_blurred, cv2.COLOR_RGB2BGR)

    # HSV is for hue saturation value i.e. for filtering by intensity
    # now, filtering by color
    # defining redness color range
    min_red = np.array([0, 100, 80])
    max_red = np.array([10, 256, 256])

    mask1 = cv2.inRange(image_blurred_hsv, min_red, max_red)

    #filtering by brightness
    min_red2 = np.array([170, 100, 80])
    max_red2 = np.array([180, 256, 256])

    mask2 = cv2.inRange(image_blurred_hsv, min_red2, max_red2)

    #combining both masks

    mask = mask1 + mask2

    # Segmentation - sepearting strawberry

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)

    # Find biggest Strawberry
    big_straw_contour, mask_straw = find_biggest_contour(mask_clean)

    # Overlay the masks that we created on image
    overlay = overlay_mask(mask_clean, image)

    # Circle biggest strawberry
    circled = circle_contour(overlay, big_straw_contour)

    show(circled)

    # Convert back to original color scheme

    bgr = cv2.cvtColor(circled, cv2.COLOR_RGB2BGR)

    return bgr
 

# Main function
#read image
 
image = cv2.imread('/mnt/Work/PycharmProjects/OpenCV_codes/berry.jpg')
result = findStrawberry(image)
# write new image
cv2.imwrite('/mnt/Work/PycharmProjects/OpenCV_codes/berry2.jpg', result)