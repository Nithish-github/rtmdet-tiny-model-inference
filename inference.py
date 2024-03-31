from mmdet.apis import init_detector, inference_detector
import cv2
import os
import json
import torch
import random
import numpy as np
import supervision as sv
import time

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CONFIG_PATH = 'rtmdet_tiny_8xb32-300e_coco.py'
WEIGHTS_PATH = 'rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'

model = init_detector(CONFIG_PATH, WEIGHTS_PATH, device=DEVICE)

VIDEO_PATH = "input.mp4"
OUTPUT_VIDEO_PATH = "output_video.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

box_annotator = sv.BoxAnnotator()

total_processing_time = 0  # Initialize total processing time
total_frames = 0  # Initialize total frames processed
max_processing_time = 0  # Initialize maximum processing time
max_processing_time_frame_number = 0  # Initialize frame number with maximum processing time
current_frame_number = 0  # Initialize current frame number

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    start_time = time.time()  # Start time for frame processing
    
    result = inference_detector(model, frame)
    detections = sv.Detections.from_mmdetection(result)
    
    # NMS suppression
    detections = detections[detections.confidence > 0.3].with_nms()
    
    # Annotate each frame
    annotated_frame = box_annotator.annotate(frame, detections)

    end_time = time.time()  # End time for frame processing
    processing_time = end_time - start_time  # Time taken to process the frame
    
    total_processing_time += processing_time  # Accumulate total processing time
    total_frames += 1  # Increment total frames processed
    
    if processing_time > max_processing_time:
        max_processing_time = processing_time  # Update maximum processing time
        max_processing_time_frame_number = current_frame_number  # Update frame number with maximum processing time
    
    processing_time_text = f"Rtmdet_tiny_model_inference"
    
    # Display processing time
    # cv2.putText(annotated_frame, processing_time_text, (500, 50), cv2.FONT_HERSHEY_SIMPLEX,2, (255, 0, 0), 2)
    
    # Write the frame to the output video
    out.write(annotated_frame)

    # Display the annotated frame
    cv2.imshow("Display", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    current_frame_number += 1  # Increment frame number

cap.release()
out.release()
cv2.destroyAllWindows()

average_processing_time = total_processing_time / total_frames
print(f"Total processing time: {total_processing_time:.2f} seconds")
print(f"Average processing time per frame: {average_processing_time:.2f} seconds")
print(f"Maximum processing time: {max_processing_time:.2f} seconds, Frame number: {max_processing_time_frame_number}")
