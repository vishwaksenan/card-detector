import cv2
import numpy as np

#image read section
frame = cv2.imread('sample-images/test.png')
frame = cv2.resize(frame,(200,400))

#color manipulation section - channel change
gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

#image-thresholding
_,threshold = cv2.threshold(gray,125,255,cv2.THRESH_BINARY_INV)
contours,_ = cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#finding the card boundaries
# edges = cv2.Canny(gray,200,200)

#finding the card number and card suite
card = cv2.imread('training_images/card/8.png',0)
suite = cv2.imread('training_images/suite/hearts.png',0)
card_size_w,card_size_h = card.shape[::-1]
suite_size_w,suite_size_h = suite.shape[::-1]

#matching the template for the card 
res1 = cv2.matchTemplate(threshold,card,cv2.TM_CCOEFF)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res1)
card_top_left = max_loc
card_bottom_right = (card_top_left[0]+card_size_w, card_top_left[1]+card_size_h)
cv2.rectangle(frame,card_top_left, card_bottom_right,(0,0,255),3)
cv2.putText(frame,"8 of",(card_top_left[0]+card_size_w + 10, card_top_left[1]+card_size_h),0,cv2.FONT_HERSHEY_PLAIN,(255,0,255),4)

#matching the template for the suit
res2 = cv2.matchTemplate(threshold,suite,cv2.TM_CCOEFF)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res2)
card_top_left = max_loc
card_bottom_right = (card_top_left[0]+card_size_w, card_top_left[1]+card_size_h)
cv2.rectangle(frame,card_top_left,card_bottom_right,(0,255,0),3)
cv2.putText(frame,"hearts",(card_top_left[0]+card_size_w + 10, card_top_left[1]+card_size_h),0,cv2.FONT_HERSHEY_PLAIN,(255,0,255),4)

#show image section
# cv2.drawContours(frame,contours,-1,(255,255,0),3)
cv2.imshow('actual_frame',frame)
cv2.waitKey(0)
cv2.destroyAllWindows()