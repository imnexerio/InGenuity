import cv2
import numpy as np
import pytesseract
import time

video_capture = cv2.VideoCapture(0)

plate_no='unrecognised'
unk_per_list=[]
entry_time_list=[]
while True:
    #print('Trying to deetect number plate')
    time.sleep(0.4)
    frame = video_capture.read()[1]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    edged = cv2.Canny(gray, 170, 200)

    cnts= cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:30] 
    NumberPlateCnt = None 

    count = 0
    for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:  
                NumberPlateCnt = approx 
                break
    # Masking the part other than the number plate
    mask = np.zeros(gray.shape,np.uint8)
    got_plate = cv2.drawContours(mask,[NumberPlateCnt],0,255,-1)
    got_plate = cv2.bitwise_and(frame,frame,mask=mask)
    #cv2.namedWindow("Final_image",cv2.WINDOW_NORMAL)

    # Configuration for tesseract
    config = ('-l eng --oem 1 --psm 3')
    pytesseract.pytesseract.tesseract_cmd= r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    # Run tesseract OCR on image
    plate_no = pytesseract.image_to_string(got_plate, lang='eng')

    # Print recognized text
    print(plate_no)
    cv2.imshow('Number plate', got_plate)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
