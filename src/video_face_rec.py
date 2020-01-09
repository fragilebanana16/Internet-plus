import os
import sys
import cv2
import numpy as np
def generate():
    face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')
    camera = cv2.VideoCapture(0)
    count = 0
    while(True):
        ret, frame = camera.read()
        
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:#could be many faces
                img = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                f = cv2.resize(gray[y:y+h,x:x+w], (200,200))#left-up x,y right-down x+w,y+w . in numpy y is first
                cv2.imwrite('faces/fl/%s.pgm'%str(count),f)#me文件夹一定要先存在，否则写入不了
                count += 1
               
            cv2.imshow('frame!',frame)
            if count > 100:
                break
            if cv2.waitKey(100) & 0xff == ord('q') :
                break
    camera.release()
    cv2.destroyAllWindows()

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


def face_rec():
    names = ['Hurton', 'Fl', 'Jack']

    [X,y] = read_images("faces")
    y = np.asarray(y, dtype=np.int32)
    
    if len(sys.argv) == 3:
        out_dir = sys.argv[2]
    
    model = cv2.face.EigenFaceRecognizer_create()#create a model to train
    model.train(np.asarray(X), np.asarray(y))#train imgs and tags
    camera = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('cascades\haarcascade_frontalface_default.xml')
    while (True):
        read, img = camera.read()
        faces = face_cascade.detectMultiScale(img, 1.3, 5)
        for (x, y, w, h) in faces:
             
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            roi = gray[x:x+w, y:y+h]
            try:
                roi = cv2.resize(roi, (200, 200), interpolation=cv2.INTER_LINEAR)
                print (roi.shape)
                params = model.predict(roi)#predict return tag and confidence ,confidence below 4000 is good,0 represents totally matched
                print ("Label: %s, Confidence: %.2f" % (params[0], params[1]))
                cv2.putText(img, names[params[0]], (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
                if (params[0] == 0):
                    cv2.imwrite('face_rec.jpg', img)
            except:
                continue
        cv2.imshow("camera", img)
        if cv2.waitKey(100) & 0xff == ord("q"):
            break
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
##test on the tags collected
##    i = read_images('faces')
##    print(i[1])
    face_rec()
    #generate()#generate faces to be trained
