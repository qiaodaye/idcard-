import cv2
import numpy as np
import myutils


def cv_show(name,img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    img = cv2.imread('/Users/qiaoye/Desktop/信用卡识别/images/ocr_a_reference.png')


    #cv_show('moban',img)

    ref = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
   # ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv_show('gray',ref)

    ref = cv2.threshold(ref,127,255,cv2.THRESH_BINARY_INV)[1]

    #cv_show('2',ref)

    #计算轮廓，cv2.findContours 函数接受的参数为二值图，即黑白，不是黑就是白

    cnts,hierarchy = cv2.findContours(ref.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(img, cnts, -1, (0, 0, 255), 3)
    #cv_show('img', img)

    print(np.array(cnts).shape) #查看找出的轮廓数是否是对的

    boundingBoxes = [cv2.boundingRect(c) for c in cnts] #用一个最小的矩形，把找到的形状包起来x,y,h,w
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),key=lambda b: b[1][0], reverse=False))

    digits = {}

    for (i,c) in enumerate(cnts):
        (x,y,w,h) = cv2.boundingRect(c)
        roi = ref[y:y+h,x:x+w]
        roi = cv2.resize(roi,(57,88))

        digits[i] = roi


    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(9,3))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))


    image = cv2.imread('/Users/qiaoye/Desktop/信用卡识别/images/credit_card_01.png')
    #cv_show('image',image)

    image = myutils.resize(image, width=300)

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
   # cv_show('image',gray)

    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)# 礼帽操作 突出明亮的区域
    #cv_show('tophat', tophat)

    gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize = -1) #只用x比 用x和y一起的效果更好 所以只用了x
    #ksize=-1相当于用3*3的

    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))  # 对他做归一化操作
    # 归一化后要把数据改成uint8的形式
    gradX = gradX.astype("uint8")
    print(np.array(gradX).shape) #打印出gradx的shape
   # cv_show('gradx',gradX)


    # 通过闭操作（先膨胀，再腐蚀）将数字连在一起
    gradx = cv2.morphologyEx(gradX,cv2.MORPH_CLOSE,sqKernel)
    #cv_show('gradx',gradx)

    thres = cv2.threshold(gradX,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1] #这个是希望电脑自己去找到自适应的阈值，0并不大是阈值
   # cv_show('thres',thres)

    #再来一个闭操作
    thres = cv2.morphologyEx(thres,cv2.MORPH_CLOSE,sqKernel)
    kernel = np.ones((6, 6), np.uint8)
    thres = cv2.dilate(thres,kernel)
    #cv_show('thres2',thres)

    threscnts,hierarchy = cv2.findContours(thres.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    cnts = threscnts
    cur_img = image
    cv2.drawContours(cur_img,cnts,-1,(0,0,255),3)
    #cv_show('img', cur_img)

    locs = []

    # 遍历轮廓

    for (i, c) in enumerate(cnts):
    #for (i, c) in enumerate(cnts):

        (x,y,w,h) = cv2.boundingRect(c)
        #(x,y,h,w) = cv2.boundingRect(c)

        #计算这个框出来的矩形的比例，通过比例进行选择
        ar = w / float(h)
        # 选择合适的区域，根据实际任务来，这里的基本都是四个数字一组
        if ar > 2.5 and ar < 3.5:

            if (w > 50 and w < 60) and (h > 10 and h < 30):
                # 符合的留下来
                locs.append((x, y, w, h))

    locs = sorted(locs,key=lambda x:x[0], reverse = False)
    output = []

    #遍历轮廓中的每个数字
    for(i,(gx,gy,gw,gh)) in enumerate(locs):
        groupOut = []

        # 根据坐标提取每一个组
        group = gray[gy - 5:gy + gh + 5, gx - 5:gx + gw + 5]
        #cv_show('group',group)

        # 预处理
        group = cv2.threshold(group,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        #cv_show('group',group)

        # 计算每一组的轮廓
        digitCnts, hierarchy = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        digitCnts = myutils.sort_contours(digitCnts,method="left-to-right")[0]

        #计算每一组中每个数的数值

        for c in digitCnts:
            (x,y,w,h) = cv2.boundingRect(c)

            roi = group[y:y+h,x:x+w]
            roi = cv2.resize(roi,(57,88))
            #cv_show('roi',roi)

            #计算匹配的得分
            scores = []

            for (digit,digitROI) in digits.items():
                result = cv2.matchTemplate(roi,digitROI,cv2.TM_CCOEFF)
                (_, score, _, _) = cv2.minMaxLoc(result)
                scores.append(score)

            groupOut.append(str(np.argmax(scores)))

        #画出来

        cv2.rectangle(image, (gx - 5, gy - 5), (gx + gw + 5, gy + gh + 5), (0, 0, 255), 1)
        cv2.putText(image,"".join(groupOut),(gx,gy-15),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0.0,255),2)
        cv_show('img',image)