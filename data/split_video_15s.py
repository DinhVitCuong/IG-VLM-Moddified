import cv2
import os
import math
from datetime import datetime

# Input and output paths
input_video = "/workspace/data/Video/L01/L01_V010_480p.mp4"
output_dir = "/workspace/data/Sub_video/L01/"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Read the video
cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    raise Exception("Error: Could not open video file")

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Calculate number of segments (15 seconds each)
segment_duration = 15  # seconds
# num_segments = math.ceil(duration / segment_duration)
num_segments = 40
frames_per_segment = int(segment_duration * fps)


# Print video information
print(f"Total duration: {duration:.2f} seconds")
print(f"Number of sub-videos to be produced: {num_segments}")

# Print start timestamps for each sub-video
for segment in range(num_segments):
    start_timestamp = segment * segment_duration
    print(f"Sub-video {segment+1} start timestamp: {start_timestamp:.2f} seconds")

# Get base filename without extension
video_name = os.path.splitext(os.path.basename(input_video))[0]

# Process each segment
for segment in range(num_segments):
    # Create output video writer
    output_path = os.path.join(output_dir, f"{video_name}_sub{segment+1}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Calculate start and end frames for current segment
    start_frame = segment * frames_per_segment
    end_frame = min((segment + 1) * frames_per_segment, total_frames)
    
    # Set video to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Write frames to segment
    for _ in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    
    # Release the segment writer
    out.release()

# Release the input video
cap.release()