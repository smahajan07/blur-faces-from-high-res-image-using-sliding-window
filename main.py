import cv2
import numpy as np
import time
import dlib
import align_dlib

object_db = align_dlib.AlignDlib('shape_predictor_68_face_landmarks.dat')
numFaces=0
faces = []

def slidingWindow(image, stepSize, windowSize):
    for y in xrange(0, image.shape[0], stepSize):
        for x in xrange(0, image.shape[1], stepSize):
            yield (x, y, image[y:y+windowSize[1], x:x+windowSize[0]])

def checkFace(testImage,x,y):
    testImageCopy = testImage.copy()
    global numFaces,faces
    retValue = object_db.getLargestFaceBoundingBox(testImage)
    if retValue == None:
        print 'No face found'
    else:
        try:
            faces.append(retValue)
            #cv2.imwrite('face.jpg', testImage)
            #testImage = cv2.GaussianBlur(testImage[retValue.top():retValue.bottom(), retValue.left():retValue.right()], (10,10), 0)
            numFaces += 1
            sub_face = testImage[retValue.top():retValue.bottom(), retValue.left():retValue.right()]
            sub_face = cv2.GaussianBlur(sub_face, (23, 23), 30)
            if ((retValue.top() + sub_face.shape[0] < testImage.shape[0]) and(retValue.left() + sub_face.shape[1] < testImage.shape[1])):
                testImageCopy[retValue.top():retValue.top() + sub_face.shape[0],retValue.left():retValue.left() + sub_face.shape[1]] = sub_face
                cv2.imwrite('BlurredFace' + str (numFaces) + '.jpg', testImageCopy)
                result_image[y:y+testImage.shape[0],x:x+testImage.shape[1]] = testImageCopy
        except:
            pass
    print numFaces


path = '' # path of image
image = cv2.imread(path)
#image = cv2.resize(img, None, fx=0.2, fy=0.2)
result_image = image.copy()
(winW, winH) = (256, 256)


for x,y,window in slidingWindow(image, stepSize= 64, windowSize=(winH, winW)):
    if window.shape[0] != winH or window.shape[1] != winW:
        continue
    checkFace(window,x,y)
    # clone = image.copy()
    # cv2.rectangle(clone, (x,y), (x+winW, y+winH), (0,255,0), 2)
    # cv2.imshow('Sliding Window', clone)
    # cv2.waitKey(1)
    # time.sleep(0.025)

print faces
cv2.imwrite('BlurredFacesFinal.jpg', result_image) #final image

cv2.waitKey(0)
cv2.destroyAllWindows()