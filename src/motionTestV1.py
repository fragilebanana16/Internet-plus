# -*- coding: utf-8 -*-
# @Time    : 2018/11/21 16:52
# @Author  : Hurton
# @Email   : 850604714@qq.com
#
#
#  @File    : kerasTutorials.py
# @Software: PyCharm Community Edition
# @Introduction: 
# do not show the warning about AVX
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# do not show the warning about AVX


import time
from keras.models import load_model
import numpy as np
import chineseText
import datetime
import cv2

startTime = datetime.datetime.now()
emotion_classifier = load_model(
    'simple_CNN.530-0.65.hdf5')# 训练好的情绪分类器
endTime = datetime.datetime.now()
print(endTime - startTime)

emotion_text = ''
emotion_flag = '平静'
emotion_labels = {
    0: '生气',
    1: '厌恶',
    2: '恐惧',
    3: '开心',
    4: '难过',
    5: '惊喜',
    6: '平静'
}
camera = cv2.VideoCapture(0)
#img = cv2.imread("angry.jpg")
face_classifier = cv2.CascadeClassifier("cascades\haarcascade_frontalface_default.xml")
while(True):
    ret, img = camera.read()
    if ret:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(40, 40))
        color = (255, 0, 0)

        for (x, y, w, h) in faces:
            gray_face = gray[(y):(y + h), (x):(x + w)]
            gray_face = cv2.resize(gray_face, (48, 48))# 网络结构(None, 48, 48, 16)
            gray_face = gray_face / 255.0
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            emotion_probability = np.max(emotion_classifier.predict(gray_face))
            emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
            emotion = emotion_labels[emotion_label_arg]
            print(emotion_probability)
            emotion_flag = emotion

            
            time.sleep(0.08)
            emotion_probability = np.max(emotion_classifier.predict(gray_face))
            emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
            emotion = emotion_labels[emotion_label_arg]
            
            if emotion_flag == emotion:
                

                if emotion == '生气':
                    print(emotion_labels[0])# 颜色由网络输出概率改变
                    color = emotion_probability * np.asarray((255, 0, 0))
                elif emotion == '厌恶':
                    color = emotion_probability * np.asarray((0, 0, 255))
                elif emotion == '恐惧':
                    color = emotion_probability * np.asarray((255, 255, 255))
                elif emotion == '开心':
                    color = emotion_probability * np.asarray((255, 255, 0))
                elif emotion == '难过':
                    color = emotion_probability * np.asarray((0, 0, 255))
                elif emotion == '惊喜':
                    color = emotion_probability * np.asarray((0, 255, 255))
                else:
                    color = emotion_probability * np.asarray((0, 255, 0))

                print(color)
                color = [int(i) for i in color] # list元素转成int
                color = tuple(color)# 传入参数需要为tuple
                
    ##            color = color.tolist()
                
                
                #cv2.rectangle(img, (x + 10, y + 10), (x + h - 10, y + w - 10),color, 2)
    img = chineseText.cv2ImgAddText(img, emotion, 0, 410, color, 70)

    cv2.imshow("Image", img)
            
            
            
    if cv2.waitKey(100) & 0xff == ord('q'):
        break
camera.release()
cv2.waitKey(0)
cv2.destroyAllWindows()

