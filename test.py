

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

from threading import Thread, Lock
import cv2
import sys

class WebcamVideoStream :
    def __init__(self, src =0, width = 800, height = 640) :
        self.stream = cv2.VideoCapture(src)
        self.stream.set(3, width)
        self.stream.set(4, height)
        (self.grabbed, self.frame) = self.stream.read()
        self.started = False
        self.read_lock = Lock()

    def start(self) :
        if self.started :
            print ("already started!!")
            return None
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self) :
        while self.started :
            (grabbed, frame) = self.stream.read()
            self.read_lock.acquire()
            self.grabbed, self.frame = grabbed, frame
            self.read_lock.release()

    def read(self) :
        self.read_lock.acquire()
        frame = self.frame.copy()
        self.read_lock.release()
        return frame

    def stop(self) :
        self.started = False
        self.thread.join()

    def __exit__(self, exc_type, exc_value, traceback) :
        self.stream.release()

#cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# video_capture = cv2.VideoCapture(0)
video_capture = WebcamVideoStream(src=0, width=640, height=400).start()
folder = 'fac'
count = 0
print("[INFO] loading network...")
model = load_model('real_fake.model')
while True:
    count += 1
    # Capture frame-by-frame
    image1 = video_capture.read()


    gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
       gray,
       scaleFactor=1.1,
       minNeighbors=5,
        minSize=(50, 50),
    )
   

    for (x, y, w, h) in faces:
        cv2.rectangle(image1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        im = image1[y:y + h, x:x + w]
        # cv2.imwrite(folder + '/' + str(count) + '.jpg',im)
        # print (im)
    
        image = cv2.resize(im, (28, 28))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        # load the trained convolutional neural network


        # classify the input image
        (fake, real) = model.predict(image)[0]


        # build the label
        if real > fake:
	    print("Real values:",real)
	    print(type(real))
            label = "Real"
            proba = real
            label = "{}: {:.2f}%".format(label, proba * 100)
            cv2.putText(image1, label, (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

        else:
            label = "Fake"
            proba = fake
            label = "{}: {:.2f}%".format(label, proba * 100)
            cv2.putText(image1, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)


        #

	

    # Display the resulting frame
    cv2.imshow('Video', image1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.stop()
cv2.destroyAllWindows()



