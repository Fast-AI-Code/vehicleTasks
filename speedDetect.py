import numpy as np
import imutils
import dlib
import cv2
import math
import time

# all classes from ssd needed for recognition
classes = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

net = cv2.dnn.readNetFromCaffe("data/model.prototxt",
                               "data/ssd.caffemodel")  # pretrained SSD

video = cv2.VideoCapture("data/clipped.mp4")

# trackers, labels = [], []
carLoc1, carLoc2, carTracker, timer = {}, {}, {}, {
}  # initialize car locations
currentId = 0
dist = 16  #real distance - random value assumed
speed_limit = 70

while True:
    (grabbed, frame) = video.read()

    if frame is None:
        break
    frame = imutils.resize(frame, width=600)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    toDel = []  # creating sep list to prevent dict size change error
    for cid in carTracker.keys():
        trackingQuality = carTracker[cid].update(frame)
        if trackingQuality < 6:
            toDel.append(cid)

    for a in toDel:
        carTracker.pop(a, None)
        carLoc1.pop(a, None)
        carLoc2.pop(a, None)

    if len(carTracker) == 0:  # basically if empty tracker list
        (h, w) = frame.shape[:2]
        distPerPixel = dist / w
        createBlob = cv2.dnn.blobFromImage(frame, 0.007843, (w, h), 127.5)
        net.setInput(createBlob)  #using dnn
        detected = net.forward()  # updating the blob

        for i in np.arange(0, detected.shape[2]):
            conf = detected[0, 0, i, 2]

            if conf > 0.5:  # confidence of object detection
                idx = int(detected[0, 0, i, 1])
                label = classes[idx]
                if label != "car":  # only check for cars and ignore rest
                    continue

                box = detected[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                t = dlib.correlation_tracker()  # use dlib tracker
                rect = dlib.rectangle(startX, startY, endX, endY)
                t.start_track(frame, rect)
                carTracker[currentId] = t  # specific id per car
                carLoc1[currentId] = [startX, startY, endX, endY]
                currentId += 1

                # labels.append(label)
                # trackers.append(t)
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (255, 255, 0), 2)
                cv2.putText(frame, label, (startX, startY - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 2)
    else:  # if already there then -> update existing tracker
        # for (t, l) in zip(trackers, labels):
        #     quality_track = t.update(frame)

        #     pos = t.get_position()
        #     startX, startY, endX, endY = int(pos.left()), int(pos.top()), int(
        #         pos.right()), int(pos.bottom())
        #     carLoc2[currentId]= [startX, startY, endX, endY]
        #     cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 255, 0),
        #                   2)
        #     cv2.putText(frame, l, (startX, startY - 15),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 2)

        for cid in carTracker.keys():
            car = carTracker[cid].update(frame)
            pos = carTracker[cid].get_position()
            startX, startY, endX, endY = int(pos.left()), int(pos.top()), int(
                pos.right()), int(pos.bottom())
            carLoc2[currentId] = [startX, startY, endX, endY]
            cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 255, 0),
                          2)

    # # calculate speed
    for cid in carTracker.keys():
        # print(carTracker, carLoc2, carLoc1)
        try:
            dist_pixels = math.sqrt(
                math.pow(carLoc2[cid][0] - carLoc1[cid][0], 2) +
                math.pow(carLoc2[cid][1] - carLoc1[cid][1], 2))
            pixels_per_meter = 30
            dist_meters = dist_pixels / pixels_per_meter
            fps = 10
            speed = dist_meters * fps  # speed in meters per second
            cv2.putText(frame, str(speed), (int(carLoc2[cid][0]), int(carLoc2[cid][1])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 2)
            if speed> speed_limit:
                print(speed, cid)
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255),
                          2)

        except KeyError:
            pass

    cv2.imshow("det", frame)
    # print(carLoc1)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cv2.destroyAllWindows()
video.release()
