from skimage.filters import threshold_local
import numpy as np
import argparse #program to automatically generate help and usage messages and issues errors from user input
import cv2
import imutils #module that contains functions for resizing, rotating and cropping images



#constructing the argument parser and parsing arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
    help = "Path to the image to be scanned")
args = vars(ap.parse_args())



# ------  using openCV to perform edge detection ------ #



# loading the image and computing the ratio of the old height to the new height, cloning it and resizing it
image = cv2.imread(args["image"])
ratio = image.shape[0] / 500
orig = image.copy()
image = imutils.resize(image, height = 500)
#this allows for raster image processing and more accurate edge tection (resizing to 500 pixels)

# converting the image to grayscale, blurring it, and finding edges using Canny edge detection algorithm
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GuassianBlur(gray, (5,5), 0)
edged = cv2.Canny(gray, 75, 200)

#showing the original image and the edge detected image
print("Step 1: Edge detection complete")
cv2.imshow("Image", image)
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()



# ------  using openCV to find contours (detect our shape) ------ #



# finding the contours in the edged image, keeping only the largest ones and initialize the screen contour
contours = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours, key=cv2.contourArea, reverse = True) [:5]

#loop over contours
for c in contours:
    #approximate the contour
    peri = cv2.arcLength(c,True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    #if our approximated contour has 4 points, then we can assume we found our screen
    if len(approx) == 4:
        screenContour = approx
        break

# show the contour of the piece of paper
print("Step 2: Finding the contours of paper")
cv2.drawContours(image, [screenContour], -1, (0, 255, 0), 2)
cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()



# ------  using openCV to apply a perspective transform and threshold ------ #



def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return warped

# applying the four point transfrom from openCV to obtain a top-down view of the original image
warped = four_point_transform(orig, screenContour.reshape(4,2) * ratio)

# convert the warped image to grayscale, then threshold it to give the "Black and White" paper effect
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset = 10, method = "gaussian")
warped = (warped > T).astype("uint8") * 255

# show the original and scanned images
print("STEP 3: Apply perspective transform")
cv2.imshow("Original", imutils.resize(orig, height = 650))
cv2.imshow("Scanned", imutils.resize(warped, height = 650))
cv2.waitKey(0)