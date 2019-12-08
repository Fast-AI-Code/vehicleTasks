import numpy as np
import imutils
import dlib
import cv2
import time

classes = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

net = cv2.dnn.readNetFromCaffe("data/model.prototxt", "data/ssd.caffemodel")
frame = cv2.imread("data/scr.png")
frame = imutils.resize(frame, width=600)
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
(h, w) = frame.shape[:2]
createBlob = cv2.dnn.blobFromImage(frame, 0.007843, (w, h), 127.5)
net.setInput(createBlob)  #using dnn
detected = net.forward()  # updating the blob

cars_list_temp = []
cno = 1
for i in np.arange(0, detected.shape[2]):
    conf = detected[0, 0, i, 2]

    if conf > 0.5:  # confidence of object detection
        idx = int(detected[0, 0, i, 1])
        label = classes[idx]
        if label != "car":  # only check for cars and ignore rest
            continue

        box = detected[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 255, 0), 2)
        cars_list_temp.append((startX, startY, endX, endY))  #xywh
        cv2.putText(frame, str(cno), (startX, startY - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 2)
        cno += 1

# Assuming car 1 (left) wants to take a turn right now

test_car1, test_car2 = cars_list_temp[0], cars_list_temp[
    1]  # Taking 1st two cars
x_car1, x_car2 = test_car1[0], test_car2[0]  # taking x coord of both cars
# x_car1, x_car2 = test_car2[0], test_car1[0]  # taking x coord of both cars

thresh = test_car1[2] + 5  # say traffic regulation (w+thresh)

# Assuming road boundaries here are the left and right corners of the image
left_boun, right_bound = 0, w
print(x_car1, left_boun,thresh)
# to check for car 1 left turn
if x_car1 - left_boun > thresh:
    cv2.putText(frame, "car 1 go left", (int(w / 2), int(h / 2)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 2)
    print(1)
elif all([right_bound - x_car1 > thresh, x_car2 - x_car1 > thresh]):
    cv2.putText(frame, "car 1 go right", (int(w / 2), int(h / 2)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 2)
    print(2)
else:
    cv2.putText(frame, "car 1 go straight", (int(w / 2), int(h / 2)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 2)
    print(3) # change thresh to a high value

cv2.imshow("det", frame)
cv2.waitKey(0)
