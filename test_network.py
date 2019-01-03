# USAGE
# python test_network.py --model santa_not_santa.model --image images/examples/santa_01.png

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
    def __init__(self, src ="rtsp://admin:gepl@123@192.168.100.64:554/", width = 800, height = 640) :
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
        # flags=cv2.CV_HAAR_SCALE_IMAGE
    )
    # cv2.imwrite(folder + '/' + str(count) + '.jpg',image)
    # img = open(folder + '/' + str(count) + '.jpg', 'rb').read()

    for (x, y, w, h) in faces:
        cv2.rectangle(image1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        im = image1[y:y + h, x:x + w]
        # cv2.imwrite(folder + '/' + str(count) + '.jpg',im)
        # print (im)
    # image = cv2.imread(str(count) + '.jpg')

    # print (faces)
    # print (type(faces))
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


        # Draw a rectangle around the faces



        # output = imutils.resize(image, width=400)
        

	

    # Display the resulting frame
    cv2.imshow('Video', image1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.stop()
cv2.destroyAllWindows()


# construct the argument parse and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-m", "--model", required=True,
#	help="path to trained model model")
#ap.add_argument("-i", "--image", required=True,
#	help="path to input image")
#args = vars(ap.parse_args())

# load the image
#image = cv2.imread(args["image"])
#orig = image.copy()

# pre-process the image for classification
#image = cv2.resize(image, (28, 28))
#image = image.astype("float") / 255.0
#image = img_to_array(image)
#image = np.expand_dims(image, axis=0)

# load the trained convolutional neural network
#print("[INFO] loading network...")
#model = load_model(args["model"])

# classify the input image
#(fake, real) = model.predict(image)[0]

# build the label
#label = "Real" if real > fake else "Fake"
#proba = real if real > fake else fake
#label = "{}: {:.2f}%".format(label, proba * 100)

# draw the label on the image
#output = imutils.resize(orig, width=400)
#cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
#	0.7, (0, 255, 0), 2)

# show the output image
#cv2.imshow("Output", output)
#cv2.waitKey(0)
