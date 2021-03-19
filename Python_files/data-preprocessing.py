import cv2
import numpy as np
import os

def mole_detection(img_link:str, new_name:str, lower_color_bound=(0,0,0), upper_color_bound=(255,110,255)):
    """
    Take the image from the database and preprocessed
    it to get only the mole with a dark background.

    Parameters:
    img_link (str): the location of the image to load.
    new_name (str): the new location and name where the image will be saved.
    lower_color_bound (tuple): the lower bound to get colors in the image.
    upper_color_bound (tuple): the upper bound to get colors in the image.
    """
    # Take the image and convert it into grayscale and hsv colors.
    img = cv2.imread(img_link,cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Make a mask based on the hsv colors.
    mask1 = cv2.inRange(hsv, lower_color_bound, upper_color_bound)

    # Otsu's thresholding
    _,mask2 = cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Invert the two masks
    mask1 = cv2.bitwise_not(mask1)
    mask2 = cv2.bitwise_not(mask2)

    # Merge the two masks and 
    merged_mask = cv2.bitwise_and(mask1, mask2)

    # Search for contours and select the biggest one
    contours, hierarchy = cv2.findContours(merged_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key=cv2.contourArea)

    # Create a new mask for the result image
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)

    # Draw the contour on the new mask and perform the bitwise operation
    cv2.drawContours(mask, [cnt],-1, 255, -1)
    res = cv2.bitwise_and(img, img, mask=mask)

    cv2.imwrite(new_name, res)

if __name__ == '__main__':

    # directory after downloading images
    datpath = '../data/Mole_Data_Rearranged'

    dir1 = os.listdir(datpath)
    dir2 = ["0", "1"]

    for i in range(len(datpath)):
        images = os.listdir(datpath + '/' + dir2[i])
        srcp = datpath + '/' + dir2[i] + '/'

        for image in images:
            img_link = srcp + image
            new_name = "../data/Preprocessed_data/" + f"{i}" + "/" + image
            try:
                mole_detection(img_link, new_name)
            except:
                continue