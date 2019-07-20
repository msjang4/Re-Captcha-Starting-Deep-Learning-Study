from PIL import Image,ImageGrab     #pip install pillow
import pytesseract as pt #pip install pytesseract
import configparser
import os
import numpy as np
import cv2
from clipboard import grab_clipboard_img
# #Config Parser 초기화
# config = configparser.ConfigParser()
# #Config File 읽기
# config.read(os.path.dirname(os.path.realpath(__file__)) + os.sep + 'envs' + os.sep + 'property.ini')
#이미지 -> 문자열 추출
imshow = False
def show(name, mat, force = False):
        if imshow or force:
                cv2.imshow(name,mat)
def imgToStr( img, n, area_n=0, kernel_size = [(3,3),(7,7)],lang='kor',outTxtPath = None,fullPath=None):
        #다음 과정을 거친다 
        # 1. CUBIC (n)
        # 2. GRAYSCALE 
        # 3. ERODE by (3,3)
        # 4. BLUR by (5,5)
        # 5. THRESHOLD (191, 255)
        # 6. CONTOUR
        # 7. DILATE by (7,7)
        # 8. CROP
        # 9. IMAGE2STRING


   

    pt.pytesseract.tesseract_cmd = "D:/Tesseract-OCR/tesseract.exe"


        #IMG
    show('img',img)
    h,w,_= img.shape
        #CUBIC
    dsize = (n, int(h * n/w ))
    cubic = cv2.resize(img, dsize=dsize, interpolation=cv2.INTER_CUBIC)
    show("cubic",cubic)
        #GRAY
    gray=cv2.cvtColor(cubic,cv2.COLOR_RGB2GRAY)
    show("gray",gray)
        #ERODE

    if kernel_size is not None:
        kernel = np.ones(kernel_size[0], np.uint8)        

        erode = cv2.erode(gray, kernel, iterations=1)
        show("erode",erode)
    else :
        erode = gray    
        #BLUR

    
    blur = cv2.GaussianBlur(erode, (5,5),0)
    show("blur",blur)

    #THRESHOLD
    ret, thr = cv2.threshold(blur, 191, 255, cv2.THRESH_BINARY_INV)
    
    show('thr',thr)
    

        # CONTOUR
    _, contours, _ = cv2.findContours(thr, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour =thr.copy()

    h,w= thr.shape
    left, right, top, bottom = w,0, h, 0
    for i in range(0, len(contours)-1):
        x,y,w,h = cv2.boundingRect(contours[i])
        if w > n/6:
            continue
        if x < left:
                left = x
        if x+w > right:
                right = x+w
        if y < top:
                top = y
        if y+h > bottom:
                bottom = y+h
        cv2.rectangle(contour,(x,y),(x+w,y+h), (127,0,0), 1)
    # print(left, top, right-left, bottom-top)
    cv2.rectangle(contour, (left,top), (right, bottom),(127,0,0),3)
    show('contour',contour)

        #DILATE
    kernel = np.ones(kernel_size[1], np.uint8)        
    dilate = cv2.erode(thr, kernel, iterations=1)   
    show("dilate",dilate)
    





        #CROP
    h,w= thr.shape
    margin = int(n*0.025)
    cropped = dilate[max(0,top-margin):min(h,bottom+margin),max(0,left-margin):min(w,right+margin)]
    show('cropped',cropped)
    
    if area_n:
        h,w = cropped.shape
        dsize = (area_n, int(h * area_n/w ))
        area = cv2.resize(cropped, dsize=dsize, interpolation=cv2.INTER_AREA)
        show('area',area)
        outText = pt.image_to_string(Image.fromarray(area), lang=lang)
    
    else:
        outText = pt.image_to_string(Image.fromarray(cropped), lang=lang)
    
    
    
    # print('l r t b :', left, right, top, bottom)
    # print('cropeed wait keys...')
    
    # cv2.waitKey(0)
    if outTxtPath is not None:
        strToTxt(outTxtPath, outText)
        
    return outText

#문자열 -> 텍스트파일 개별 저장
def strToTxt(txtPath, outText):
    with open(txtPath , 'w', encoding='utf-8') as f:
        f.write(outText)

#메인 시작
if __name__ == "__main__":
    imshow = True
    print(imgToStr(grab_clipboard_img(), 1500,500))
    cv2.waitKey(0)