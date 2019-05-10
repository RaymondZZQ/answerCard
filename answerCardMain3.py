#coding=utf-8
'''
file answerCardMain.py
识别答题卡主程序
'''

import cv2
import numpy as np
import os
import math
from matplotlib import pyplot as plt
from PIL import Image,ImageColor,ImageFont,ImageDraw

BASE_DIR = os.getcwd()
path_card = os.path.join(BASE_DIR,"card")
path_tmp = os.path.join(BASE_DIR,"tmp")


##在图片上添加文字
def drawXH(loc,pre):
    draw = ImageDraw.Draw(img)
    myfont = ImageFont.truetype(BASE_DIR + r'/ziti/DejaVuSans.ttf', size=40)
    mycolor = ImageColor.colormap.get('blue')
    draw.text((loc[0] , loc[1]), str(pre), font=myfont, fill=mycolor)
    return img

def drawXH2(loc,pre):
    draw = ImageDraw.Draw(img)
    myfont = ImageFont.truetype(BASE_DIR + r'/ziti/DejaVuSans.ttf', size=27)
    mycolor = ImageColor.colormap.get('blue')
    answer= ["A","B","C","D"]
    draw.text((loc[0] , loc[1]), answer[pre], font=myfont, fill=mycolor)
    return img

def StuIDRec(stuBlock,block_2,block_5,rDraw):
    ker1 = np.ones((2, 2), np.uint8)
    stuBlock1 = cv2.morphologyEx(stuBlock, cv2.MORPH_OPEN, ker1)

    ker2 = np.ones((20, 20), np.uint8)
    stuBlock2 = cv2.morphologyEx(block_2, cv2.MORPH_OPEN, ker2)

    _, contours1, hierarchy = cv2.findContours(stuBlock2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours1[0])

    roi = stuBlock1[y:y+h,x-w:x+2*w]
    roi2 = block_5[y:y+h,x-w:x+2*w]

    cN = np.mean(roi, axis=0)
    cN[cN<150] = 0
    cN[cN>=150] = 1

    loca1 = []
    loca2 = []
    flag = True
    for i in range(len(cN)-1):
        flag1 = (cN[i] == 1 and cN[i+1] == 0)#下降沿
        flag2 = (cN[i] == 0 and cN[i+1] == 1)#上升沿

        if flag1 and flag:#下降沿
            loca1.append(i)
            flag = False
        if flag2 and not flag:
            loca2.append(i)
            flag = True

    num= []
    step = int((h-2) / 10)
    for j in range(len(loca1)):
        for k in range(0,10):
            blockThis = roi2[k*step+5:(k+1)*step+5,loca1[j]:loca2[j]]
            if np.sum(blockThis ) > 5000:
                num.append(k)
                cDraw = loca1[j]+50
                drawXH([cDraw+100,rDraw+30,],k)
                break

    return num

def ExamIDRec(block_2,block_5,rDraw,cBias):

    ker2 = np.ones((20, 15), np.uint8)
    stuBlock2 = cv2.morphologyEx(block_2, cv2.MORPH_OPEN, ker2)

    _, contours1, hierarchy = cv2.findContours(stuBlock2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    loca1 = []
    locay = []
    locah = []
    th1 = 4
    for i in range(len(contours1)):
        x, y, w, h = cv2.boundingRect(contours1[i])
        if h > 20 and w > 20:
            locay.append(y)
            locah.append(h)
            loca1.append(x+th1)
            loca1.append(x+w-th1)

    loca1 = sorted(loca1)
    hmean = np.mean(locah)
    ymean = int(np.mean(locay))

    num = []
    step =  int((hmean) / 10)
    for j in range(len(loca1)-1):
        for k in range(10):
            blockThis=block_5[k*step+ymean:(k+1)*step+ymean,loca1[j]:loca1[j+1]]
            if np.sum(blockThis ) > 5000:
                num.append(k)
                cDraw = loca1[j] + cBias-80
                drawXH([ cDraw+80,rDraw + 40], k)
                break
    return num

def answerBlock(blockThis,block5This,xBias,yBias):
    rN = np.mean(blockThis, axis=1)
    cN = np.mean(blockThis, axis=0)
    th1 = 10
    loca = [0,len(rN),0,len(cN)]
    for r1 in range(len(rN)):
        if rN[r1] > th1 :
            loca[0] = r1
            for r2 in range(len(rN)-1,-1,-1):
                if rN[r2] > th1:
                    loca[1] = r2
                    break
            break
    for c1 in range(len(cN)):
        if cN[c1] > th1 :
            loca[2] = c1
            for c2 in range(len(cN)-1,-1,-1):
                if cN[c2] > th1:
                    loca[3] = c2
                    break
            break
    roi5This = block5This[loca[0]:loca[1],loca[2]:loca[3]]

    step = (loca[1]-loca[0]+5) // 5
    step2 = (loca[3]-loca[2]+5) // 5

    th2 = 3000
    num = []
    for i in range(0,5):
        for j in range(0,4):
            thisRoiMin = roi5This[i*step:i*step+step,(j+1)*step2:(j+1)*step2+step2]
            if np.sum(thisRoiMin) > th2 :
                num.append(j)
                cDraw = xBias
                rDraw = i*step + yBias
                drawXH2([ cDraw+185,rDraw+5], j)
                break
    return num


def AnswerRec(ansBlock,blockIm5,xBias,yBias):

    ker1 = np.ones((3, 3), np.uint8)
    ansBlock = cv2.morphologyEx(ansBlock, cv2.MORPH_OPEN, ker1)
    numBlock = ansBlock.shape[0] // 130
    step = int((ansBlock.shape[0]+20) / numBlock)
    re = cv2.cvtColor(ansBlock,cv2.COLOR_GRAY2BGR)
    th = 6
    num = []
    for i in range(0,numBlock):
        re = cv2.cvtColor(ansBlock, cv2.COLOR_GRAY2BGR)
        re[i * step:i * step + step-th,:,2]=255
        blockThis = ansBlock[i*step:i*step+step-th]
        block5This = blockIm5[i*step:i*step+step-th]
        numThis = answerBlock(blockThis,block5This,xBias,yBias+i*step)
        num += numThis
    return num

def rotate(angle,img,borderValue1):
    width = img.shape[1]
    height = img.shape[0]
    degree = angle
    heightNew = int(width * abs(math.sin(math.radians(degree))) + height * abs(math.cos(math.radians(degree))))  # 这个公式参考之前内容
    widthNew = int(height * abs(math.sin(math.radians(degree))) + width * abs(math.cos(math.radians(degree))))
    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
    matRotation[0, 2] += (widthNew - width) / 2   # 因为旋转之后,坐标系原点是新图像的左上角,所以需要根据原图做转化
    matRotation[1, 2] += (heightNew - height) / 2
    img = cv2.warpAffine(img, matRotation, (int(widthNew), int(heightNew)), borderValue=borderValue1)###todo:bordweValue需要修改
    return img


def cardProcess(img):
    print(img.shape)
    _,im1 = cv2.threshold(img,220,255,cv2.THRESH_BINARY_INV)
    _, im2 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV)
    im3 = cv2.adaptiveThreshold(img, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C ,cv2.THRESH_BINARY_INV,31,5)
    ker1 = np.ones((3,8),np.uint8)
    im4 = cv2.morphologyEx(im3,cv2.MORPH_OPEN,ker1)
    im5 = np.zeros_like(im4)
    im_result = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    _,contours1, hierarchy = cv2.findContours(im4,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for i in range(0,len(contours1)):
        x,y,w,h = cv2.boundingRect(contours1[i])

        block=im4[y:y+h,x:x+w]
        if 6<h<30 and 10<w<40:
            im5[y:y + h, x:x + w] = im4[y:y + h, x:x + w]
            im_result[y:y + h, x:x + w, 2] = 255

    _, contours2, hierarchy = cv2.findContours(im3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(0, len(contours2)):
        x, y, w, h = cv2.boundingRect(contours2[i])

        block = im3[y:y + h, x:x + w]
        block_2 = im1[y:y + h, x:x + w]
        block_5 = im5[y:y + h, x:x + w]
        #answer1
        if 800<h<900 and 180<w<250:
            block = block[5:-5, 12:-12]
            block_5 = block_5[5:-5, 12:-12]
            AnswerRec(block,block_5,x,y)

        # answer2
        if 400 < h < 500 and 180 < w < 250:
            block = block[5:-5, 12:-12]
            block_5 = block_5[5:-5, 12:-12]
            AnswerRec(block,block_5,x,y)

        #stuId
        if 280<h<350 and 440<w<540:
            stuID = StuIDRec(block,block_2,block_5,y)

        #examID
        if 280<h<350 and 300<w<360:
            ExamID = ExamIDRec( block_2, block_5,y,x)
    return 0

os.chdir(path_card)
imthis = '1.jpg'
im = cv2.imread(imthis,cv2.IMREAD_GRAYSCALE)
img = Image.open(imthis)
os.chdir(path_tmp)
cardProcess(im)
img.save("re.jpg")