import cv2
import numpy as np
import time
from collections import defaultdict
vehicle_counts = defaultdict(int)
# Load Yolo
net = cv2.dnn.readNetFromDarknet("cfg/yolov3-tiny.cfg", "weights/yolov3-tiny.weights")

classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
try:
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
except IndexError:
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
vehicle_category_map = {
    
    
    'bus': 'Small Bus',  # Can split this later into small/big bus based on size
    'truck': 'LCV',  # Light Commercial Vehicle; further classification by heuristics
    'ambulance': 'Ambulance',
    'train': 'Multi-Axle Vehicles'  # Example, you can use further classes here
}

def classify_vehicle_based_on_size(w, h, label):
    if label == "bus":
        # Example heuristic for Small Bus and Big Bus
        if w * h > 50000:
            return "Big Bus"
        else:
            return "Small Bus"
    if label == "ambulance":
        # Heuristic to classify trucks based on bounding box size
        if 50000<w * h < 60000:
            return "Multi-Axle Truck"
        
    if label == "truck":
        # Heuristic to classify trucks based on bounding box size
        if w * h > 80000:
            return "Multi-Axle Truck"
        elif w * h > 60000:
            return "3-Axle Truck"
        else:
            return "2-Axle Truck"
    
    return vehicle_category_map.get(label, label)
# Loading image
cap = cv2.VideoCapture("855981-hd_1920_1080_30fps.mp4")
#cap = cv2.VideoCapture("bangalore-city-center-day-time-traffic-street-panorama-4k-india.mp4")


font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0
while True:
    _, frame = cap.read()
    frame_id += 1

    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.88:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[3] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 1.8)
                y = int(center_y - h / 1.8)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            # Classify vehicle into custom category
            vehicle_category = classify_vehicle_based_on_size(w, h, label)
            vehicle_counts[vehicle_category] += 1
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 2, color, 2)

    y_offset = 80  # Starting vertical position
    for category, count in vehicle_counts.items():
        cv2.putText(frame, f"{category}: {count}", (10, y_offset),
                    font, 1.2, (255, 255, 255), 2)
        y_offset += 30



    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 50), font, 2, (0, 0, 0), 3)
    cv2.imshow("Image", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

