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



def read_images(path, sz=None):
    c = 0
    X,y = [], []
    for dirname, dirnames, filenames in os.walk(path):#iter all dirname or filename in it
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    if (filename == ".directory"):
                        continue
                    filepath = os.path.join(subject_path, filename)
                    print(filepath)
                    im = cv2.imread(os.path.join(subject_path, filename), cv2.IMREAD_GRAYSCALE)
                    
                    if (im is None):
                        print ("image " + filepath + " is none")
                    # resize to given size (if given)
                    if (sz is not None):
                        im = cv2.resize(im, sz)
                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)
                except IOError (errno, strerror):
                    print ("I/O error({0}): {1}".format(errno, strerror))
                except:
                    print ("Unexpected error:", sys.exc_info()[0])
                    raise
            c = c+1
    return [X,y]


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
names = ['Hurton', 'Fl', 'Jack']

[X,y] = read_images("faces")
y = np.asarray(y, dtype=np.int32)



model = cv2.face.EigenFaceRecognizer_create()#create a model to train
model.train(np.asarray(X), np.asarray(y))#train imgs and tags

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
                
                img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                roi = gray[x:x+w, y:y+h]
                try:
                    roi = cv2.resize(roi, (200, 200), interpolation=cv2.INTER_LINEAR)
                    print (roi.shape)
                    params = model.predict(roi)#predict return tag and confidence ,confidence below 4000 is good,0 represents totally matched
                    print ("Label: %s, Confidence: %.2f" % (params[0], params[1]))
                    cv2.putText(img, names[params[0]], (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
                except:
                    continue
                #cv2.rectangle(img, (x + 10, y + 10), (x + h - 10, y + w - 10),color, 2)
    img = chineseText.cv2ImgAddText(img, emotion, 0, 410, color, 70)

    cv2.imshow("Image", img)
            
    
            
    if cv2.waitKey(100) & 0xff == ord('q'):
        break


camera.release()
cv2.waitKey(0)
cv2.destroyAllWindows()

