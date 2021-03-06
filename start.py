import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle,collections
from numpy.linalg import norm
import sys
# hog=cv2.HOGDescriptor()


bin_n=32
winSize = (16,16)
blockSize = (16,16)
blockStride = (4,4)
cellSize = (8,8)
nbins = 9
derivAperture = 1
winSigma = 4.
histogramNormType = 0
L2HysThreshold = 2.0000000000000001e-01
gammaCorrection = 0
nlevels = 32
hog1 = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
hog2=cv2.HOGDescriptor()
#compute(img[, winStride[, padding[, locations]]]) -> descriptors
winStride = (8,8)
padding = (8,8)
def hog(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bin_n = 16 # Number of bins
    bin = np.int32(bin_n*ang/(2*np.pi))

    bin_cells = []
    mag_cells = []

    cellx = celly = 8

    for i in range(0,img.shape[0]/celly):
        for j in range(0,img.shape[1]/cellx):
            bin_cells.append(bin[i*celly : i*celly+celly, j*cellx : j*cellx+cellx])
            mag_cells.append(mag[i*celly : i*celly+celly, j*cellx : j*cellx+cellx])

    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)

    # transform to Hellinger kernel
    eps = 1e-7
    hist /= hist.sum() + eps
    hist = np.sqrt(hist)
    hist /= norm(hist) + eps

    return hist
# def hog(img):
#     gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
#     gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
#     mag, ang = cv2.cartToPolar(gx, gy)
#     bins = np.int32(bin_n*ang/(2*np.pi))
#     # print bins.shape# quantizing binvalues in (0...16)
#     bin_cells = bins[:7,:7], bins[7:14,:7], bins[14:21,:7], bins[21:28,:7],bins[:7,7:14], bins[7:14,7:14], bins[14:21,7:14], bins[21:28,7:14],bins[:7,14:21], bins[7:14,14:21], bins[14:21,14:21], bins[21:28,14:21],bins[:7,21:28], bins[7:14,21:28], bins[14:21,21:28], bins[21:28,21:28]
#     mag_cells = mag[:7,:7], mag[7:14,:7], mag[14:21,:7], mag[21:28,:7],mag[:7,7:14], mag[7:14,7:14], mag[14:21,7:14], mag[21:28,7:14],mag[:7,14:21], mag[7:14,14:21], mag[14:21,14:21], mag[21:28,14:21],mag[:7,21:28], mag[7:14,21:28], mag[14:21,21:28], mag[21:28,21:28]
#     hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
#     hist = np.hstack(hists)     # hist is a 64 bit vector
#     return hist
def largeCon(img):
    # img = cv2.bitwise_not(img)

    new_img = np.zeros_like(img)

    # step 1
    for val in np.unique(img)[1:]:                                      # step 2
        mask = np.uint8(img == val)                                     # step 3
        labels, stats = cv2.connectedComponentsWithStats(mask, 8)[1:3]  # step 4
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])      # step 5
        new_img[labels == largest_label] = val

    # cv2.imshow('large', new_img)
    # cv2.waitKey(0)
    return new_img

def rotateImage(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

e1 = cv2.getTickCount()
img1=cv2.imread(str(sys.argv[1]))
img1=cv2.resize(img1,(4096,4096))
# img1=rotateImage(img1,45
img=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
# print img.shape
can=img

# can=cv2.resize(img,(28,28))
# can=hog.compute(can)
# print can

# img = cv2.imread('dave.jpg',0)
# img = cv2.medianBlur(img,5)
# img = cv2.GaussianBlur(img,(3,3),0)
# img = cv2.bilateralFilter(img,9,75,75)

# ret,th2 = cv2.threshold(img,110,255,cv2.THRESH_BINARY)
# th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
#             cv2.THRESH_BINARY,11,2)
# th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#             cv2.THRESH_BINARY,21,1)

# blur = cv2.GaussianBlur(img,(5,5),0)
# ret4,th4 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# can=cv2.Canny(img,50,100)

# ret,th1 = cv2.threshold(can,200,255,cv2.THRESH_BINARY)
can = cv2.adaptiveThreshold(can,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,101,11)
# kernel = np.ones((3,3),np.uint8)
# can=cv2.erode(can,kernel,iterations=1)
# can=can[300:500,400:600]
can1=can
# kernel = np.ones((3,3),np.uint8)
# can=cv2.dilate(can,kernel,iterations=1)
# can=cv2.erode(can,kernel,iterations=1)
# #
# th1=cv2.resize(can,(1024,1024))
# # th1=cv2.medianBlur(th1,3)
# # kernel = np.ones((1,1),np.uint8)
# # th1 = cv2.morphologyEx(can,cv2.MORPH_OPEN,kernel)
#

# params=cv2.SimpleBlobDetector_Params()
# params.minThreshold = 1;
# params.maxThreshold = 200;
#
# # Filter by Area.
# params.filterByArea = True
# params.minArea = 15
#
# # Filter by Circularity
# params.filterByCircularity = True
# params.minCircularity = 0.1
#
# # Filter by Convexity
# params.filterByConvexity = True
# params.minConvexity = 0.87
#
# # Filter by Inertia
# params.filterByInertia = True
# params.minInertiaRatio = 0.01
#
#
# print can.shape
# detector=cv2.SimpleBlobDetector_create(params)
# keypoints = detector.detect(can)

# # Draw detected blobs as red circles.
# # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
# im_with_keypoints = cv2.drawKeypoints(can, keypoints, np.array([]), (0, 0, 255),
#                                       cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# # Show keypoints
# cv2.imshow("Keypoints", im_with_keypoints)
# cv2.waitKey(0)
#

# _,contours, hierarchy = cv2.findContours(can,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#
# cv2.drawContours(can, contours, -1, (0,255,0), 3)

# output = cv2.connectedComponentsWithStats(can, 4, cv2.CV_32S)
# print output

#
# cv2.imshow('jddjz',can1)
# cv2.waitKey(0)

im2, contours, hierarchy = cv2.findContours(can,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# _,contours1,hierarchy1=cv2.findContours(can,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
# print contours
#

try: hierarchy = hierarchy[0]
except: hierarchy = []

height, width = can1.shape
min_x, min_y = width, height
# print min_x,min_y
max_x = max_y = 0

ans=[]

# computes the bounding box for the contour, and draws it on the frame,
for contour, hier in zip(contours, hierarchy):
    # print contour.shape,hier
    # k = cv2.isContourConvex(contour)
    # print k
    (x,y,w,h) = cv2.boundingRect(contour)
    # print x,y,w,h/home/aviformat/Documents/intern-work
    min_x, max_x = min(x, min_x), max(x+w, max_x)
    min_y, max_y = min(y, min_y), max(y+h, max_y)
    if w>100 or h>100:
        continue
    if w>1.5*h or h>1.5*w:
        continue
    if w > 25 and h > 25:
        b=[]
        b.append(x)
        b.append(y)
        b.append(w)
        b.append(h)
        ans.append(b)
        # print x,y,w,h
        cv2.rectangle(img1, (x-1,y-1), (x+w+1,y+h+1), (255,0,0), 2)

    # if max_x - min_x > 0 and max_y - min_y > 0:
    #     cv2.rectangle(img1, (min_x, min_y), (max_x+2, max_y+2), (0,0,255), 1)
# for con, hr in zip(contours1, hierarchy1):
#     (x, y, w, h) = cv2.boundingRect(con)
#     min_x, max_x = min(x, min_x), max(x+w, max_x)
#     min_y, max_y = min(y, min_y), max(y+h, max_y)
#     if w>100 or h>100:
#         continue
#     if w > 6 and h > 6:
#         b=[]
#         b.append(x)
#         b.append(y)
#         b.append(w)
#         b.append(h)
#         ans.append(b)
#         # print x,y,w,h
#         cv2.rectangle(img1, (x-1,y-1), (x+w+1,y+h+1), (0,0,255), 1)
# cv2.imshow('blah',img1[0:598,6:900])
# cv2.waitKey(0)
#
# th1=cv2.resize(can,(2048,2048))
# # print th1
# cv2.imshow('1',img1[2009:2025,236:257])
# cv2.imshow('2',img1[2008:2024,491:501])
# cv2.imshow('3',img1[2008:2024,479:488])
# cv2.waitKey(0)
# kernel1=np.ones((1,51),np.uint8)
# bak=cv2.erode(th1,kernel1,iterations=1)
# kernel2=np.ones((51,1),np.uint8)
# bak2=cv2.erode(th1,kernel2,iterations=1)
# # #can=cv2.Canny(img,15000,20000,True)
# #
# th1=th1-(bak+bak2)
ans=sorted(ans, key=lambda element: (element[1]))
line_no=0
for i in range(len(ans)):
    if i ==0:
        ans[i].append(0)
        continue
    if abs(ans[i][1]-ans[i-1][1])>40:
        line_no+=1
    ans[i].append(line_no)

# print ans
ans=sorted(ans, key=lambda element: (element[4],element[0])) #key = y * 3000 + x.
# print len(ans)
# cv2.imshow('sdsd',img1[:224,75:106])
# cv2.imshow('blah',img1)
cv2.imwrite('eg/'+sys.argv[1],img1)
# cv2.waitKey(0)
cv2.destroyAllWindows()
e3 = cv2.getTickCount()

clf = pickle.load(open('hog900SVM200C100.sav', 'rb'))
e4=cv2.getTickCount()
# print "1"
# print (e4-e3)/cv2.getTickFrequency()
list=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

f=open("output.txt",'w+')

count_word=0
count_0=0
count_90=0
count_180=0
count_270=0

rang=len(ans)
if rang>100:
    rang=100
for i in range(rang):
    im=can1[ans[i][1]-3:ans[i][1]+ans[i][3]+3,ans[i][0]-3:ans[i][0]+ans[i][2]+3]
    # im = cv2.resize(im, (64, 128))
    im = largeCon(im)
    # im = cv2.bitwise_not(im)
    im = cv2.copyMakeBorder(im, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    # im = largeCon(im)
    # im = cv2.GaussianBlur(im, (5, 5), 0)
    # kernel = np.ones((3, 3), np.uint8)
    # im=cv2.erode(im,kernel,iterations=1)
    # im = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 5)
    # im=im.ravel()
    # print collections.Counter(im)
    # im = cv2.bitwise_not(im)
    im1=rotateImage(im,90)
    im2=rotateImage(im,180)
    im3=rotateImage(im,270)

    # im=cv2.resize(im,(32,64))
    # cv2.imshow('sdgdsg',im)
    # cv2.waitKey(0)
    # print im.shape
    # blah = collections.Counter(im[0])
    # print blah

    # im = cv2.adaptiveThreshold(im, 255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 27, 0)
    # ret, labels = cv2.connectedComponents(im)
    # print labels
    # im=largeCon(im)

    # cv2.imshow('sgsdgs',im2)
    # cv2.waitKey(0)
    # kernel=np.ones((3,3),np.uiqqions=2)
    # cv2.imshowq(0)

    # kernel = np.ones((3,3),np.uint8)
    # im=cv2.dilate(im,kernel,iterations=1)


    im=cv2.resize(im,(32,32))
    im1 = cv2.resize(im1, (32, 32))
    im2 = cv2.resize(im2, (32, 32))
    im3 = cv2.resize(im3, (32, 32))

    kernel = np.ones((3, 3), np.uint8)
    im=cv2.erode(im,kernel,iterations=1)
    im = cv2.GaussianBlur(im, (3, 3), 0)
    im = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 63, 0)

    im1 = cv2.GaussianBlur(im1, (3, 3), 0)
    im1 = cv2.adaptiveThreshold(im1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 63, 0)

    im2 = cv2.GaussianBlur(im2, (3, 3), 0)
    im2 = cv2.adaptiveThreshold(im2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 63, 0)

    im3 = cv2.GaussianBlur(im3, (3, 3), 0)
    im3 = cv2.adaptiveThreshold(im3, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 63, 0)

    # im1=rotateImage(im,90)
    # im2=rotateImage(im,180)
    # # im3=rotateImage(im,270)
    #
    cv2.imshow('saf',im)
    cv2.waitKey(0)
    # cv2.imshow('sff',im2)
    # cv2.waitKey(0)

    im=hog1.compute(im,winStride,padding)
    im1=hog1.compute(im1,winStride,padding)
    im2=hog1.compute(im2,winStride,padding)
    im3=hog1.compute(im3,winStride,padding)
    # im=np.asarray(im)
    # print im.shape
    nx,ny=im.shape
    im=im.reshape((1,nx*ny))

    nx,ny=im1.shape
    im1=im1.reshape((1,nx*ny))

    nx,ny = im2.shape
    im2 = im2.reshape((1, nx*ny))

    nx,ny= im3.shape
    im3 = im3.reshape((1, nx*ny))
    # print im

    # im=im.ravel()
    # im=im.reshape(1,-1)                                      
    # print im.shape

    ans_0=clf.predict(im)
    ans_1=clf.predict(im1)
    ans_2=clf.predict(im2)
    ans_3=clf.predict(im3)

    ans_fin=[]
    ans_fin.append(ans_0[0])
    ans_fin.append(ans_1[0])
    ans_fin.append(ans_2[0])
    ans_fin.append(ans_3[0])



    prob=clf.predict_proba(im)
    prob1=clf.predict_proba(im1)
    prob2=clf.predict_proba(im2)
    prob3=clf.predict_proba(im3)


    # print max(prob[0]),max(prob1[0]),max(prob2[0]),max(prob3[0]),

    blah=[]
    blah.append(max(prob[0]))
    blah.append(max(prob1[0]))
    blah.append(max(prob2[0]))
    blah.append(max(prob3[0]))

    print max(blah),


    index = max(xrange(len(blah)), key=blah.__getitem__)
    if index==0:# and max(blah)>0.4:
        count_0+=1
    if index==1:# and max(blah)>0.4:
        count_90+=1
    if index==2:# and max(blah)>0.4:
        count_180+=1
    if index==3:# and max(blah)>0.4:
        count_270+=1


    print str(index*90),

    # if max(prob1[0])>0.20:
    #     count_word+=1
    # print abs(abs(ans[i+1][0])-abs(ans[i][0]+ans[i][2])),
    if i==0:
        print list[ans_fin[index]],
        # f.write(list[ans_fin[index]])
    elif ans[i][4]>ans[i-1][4]:
        print "\n"
        # f.write("\n")
        print list[ans_fin[index]],
        f.write(list[ans_fin[index]])
    elif(i>0 and abs(abs(ans[i][0])-abs(ans[i-1][0]+ans[i-1][2]))<12):
        print list[ans_fin[index]],
        f.write(list[ans_fin[index]])
    else:
        print " ",
        print list[ans_fin[index]],
        f.write(" ")
        f.write(list[ans_fin[index]])
    print "\n"



    # print im

# print "\n\n"
print count_0,count_90,count_180,count_270
max_count=[]
max_count.append(count_0)
max_count.append(count_90)
max_count.append(count_180)
max_count.append(count_270)

index = max(xrange(len(max_count)), key=max_count.__getitem__)
#if index == 0:
angle=(360-(index*90))%360
print "Orientation of Given image is : "+str(angle)+" degree"
e2 = cv2.getTickCount()
time=(e2-e1)/cv2.getTickFrequency()
print "Time to execute this code is : "+str(time)
img1=cv2.resize(img1,(512,512))
img2=rotateImage(img1,index*90)

final_img = np.concatenate((img1, img2), axis=1)
cv2.imshow('final image',final_img)
cv2.waitKey(0)

#if index == 1:
 #   print "Orientation of Given image is : "+str(270)

cv2.destroyAllWindows()

