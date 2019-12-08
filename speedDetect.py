import numpy as np
import imutils
import dlib
import cv2

classes = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

net = cv2.dnn.readNetFromCaffe("data/model.prototxt", "data/ssd.caffemodel")

video = cv2.VideoCapture("data/clipped.mp4")

trackers, labels = [], []

while True:
    (grabbed, frame) = video.read()
    if frame is None:
        break

    frame = imutils.resize(frame, width=600)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if len(trackers) == 0:
        (h, w) = frame.shape[:2]
        createBlob = cv2.dnn.blobFromImage(frame, 0.007843, (w, h), 127.5)
        net.setInput(createBlob)
        detected = net.forward()

        for i in np.arange(0, detected.shape[2]):
            conf = detected[0, 0, i, 2]
            if conf > 0.5:
                idx = int(detected[0, 0, i, 1])
                label = classes[idx]
                if label != "car":
                    continue

                box = detected[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                t = dlib.correlation_tracker()
                rect = dlib.rectangle(startX, startY, endX, endY)
                t.start_track(frame, rect)

                labels.append(label)
                trackers.append(t)
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (255, 255, 0), 2)
                cv2.putText(frame, label, (startX, startY - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 2)
    else:
        for (t, l) in zip(trackers, labels):
            quality_track = t.update(frame)
            if quality_track<6:
                trackers.remove(t)
                labels.remove(l)
            pos = t.get_position()
            startX, startY, endX, endY = int(pos.left()), int(pos.top()), int(
                pos.right()), int(pos.bottom())
            cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 255, 0),
                          2)
            cv2.putText(frame, l, (startX, startY - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 2)

    cv2.imshow("det", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cv2.destroyAllWindows()
video.release()
