import cv2
import matplotlib.pyplot as plt
# import numpy as np

def detect_plate(image, face_cascade, scalefactor, min_neightbor, minsize, gaussianKernel, sigma):
    
    
    grayScale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert the image to gray scale
    gray = cv2.equalizeHist(grayScale_image) #improve the contrast of the image

    faces = face_cascade.detectMultiScale(gray, scaleFactor= scalefactor, minNeighbors= min_neightbor, minSize= minsize) # detect faces in the image
    temp = []
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2) #draw the bounding box around the face that is being detected


        face_region_gray = gray[y:y + h, x:x + w] # crop the face region
        face_region_color = image[y:y + h, x:x + w] # crop the face region
        temp.append(image) # append the image with face bounding box to the list


        blurred_face = cv2.GaussianBlur(face_region_gray, gaussianKernel, sigma) # apply gaussian blur to the eye region
        face_region_color[:, :] = cv2.cvtColor(blurred_face, cv2.COLOR_GRAY2BGR)
        
        temp.append(image) # append the image with blurred eyes to the list
    
    return temp 
    

def display_images(images):
    row_title = ["group", "none-human", "distance"] # Titles for each row
    column_title = ['original', 'face bound box','face blurred'] # Titles for each column

    fig, axes = plt.subplots(nrows=len(row_title), ncols=len(column_title), figsize=(14, 10)) # Create a grid of subplots

    for ax, col in zip(axes[0], column_title):
        ax.set_title(col, fontsize=12, fontweight='bold') # Set the title for each column
    
    for ax, row in zip(axes[:,0], row_title):
        ax.set_ylabel(row, fontsize=12, fontweight='bold', rotation=90, labelpad=10) # Set the title for each row

    for i in range(len(images)):
        plt.subplot(3,3, i+1) # Create a subplot for each image
        plt.imshow(images[i]) # Display the image
        plt.xticks([]), plt.yticks([]) # Hide tick marks
    
    return

image1 = cv2.imread('image1.png')
image2 = cv2.imread('image2.png')
image3 = cv2.imread('image4.jpg')
s1 = cv2.imread('image1.png')
s2 = cv2.imread('image2.png')
s3 = cv2.imread('image4.jpg')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") # Load the pre-trained Haar Cascade classifier for face detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml") # Load the pre-trained Haar Cascade classifier for eye detection

image_list = []

image_list.append(s1)
image = detect_plate(image1, face_cascade, scalefactor= 1.1, min_neightbor= 5, minsize= (80,80), gaussianKernel= (61,61), sigma= 40)
image_list.append(image[0])
image_list.append(image[1])

image_list.append(s2)
image = detect_plate(image2, face_cascade, scalefactor= 1.1, min_neightbor= 5, minsize= (80,80), gaussianKernel= (91,91), sigma= 60)
image_list.append(image[0])
image_list.append(image[1])

image_list.append(s3)
image = detect_plate(image3, face_cascade, scalefactor= 2, min_neightbor= 5, minsize= (80,80), gaussianKernel= (61,61), sigma= 40)
image_list.append(image[0])
image_list.append(image[1])


display_images(image_list)
plt.show()