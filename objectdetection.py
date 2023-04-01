import tensorflow_hub as hub
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf


# Loading model directly from TensorFlow Hub
detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite2/detection/1")

# Loading csv with labels of classes
labels = pd.read_csv('raspberry-pi-tensorflow/labels.csv', sep=';', index_col='ID')
labels = labels['OBJECT (2017 REL.)']

# Open a video capture object for the default camera
cap = cv2.VideoCapture('out.mp4')

while True:
    # Capture a frame from the video
    ret, frame = cap.read()

    # Convert frame to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to uint8
    rgb_tensor = tf.convert_to_tensor(rgb, dtype=tf.uint8)

    # Add dims to rgb_tensor
    rgb_tensor = tf.expand_dims(rgb_tensor , 0)

    # Detect objects in the image
    boxes, scores, classes, num_detections = detector(rgb_tensor)

    # Process outputs
    pred_labels = classes.numpy().astype('int')[0] 
    pred_labels = [labels[i] for i in pred_labels]
    pred_boxes = boxes.numpy()[0].astype('int')
    pred_scores = scores.numpy()[0]

    # Draw boxes and labels on the frame
    for score, (ymin,xmin,ymax,xmax), label in zip(pred_scores, pred_boxes, pred_labels):
        if score < 0.5:
            continue
        score_txt = f'{100 * round(score)}%'
        frame = cv2.rectangle(frame,(xmin, ymax),(xmax, ymin),(0,255,0),2)      
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, label,(xmin, ymax-10), font, 1.0, (255,0,0), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Object Detection', frame)

    # Wait for key press or 1 millisecond
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
