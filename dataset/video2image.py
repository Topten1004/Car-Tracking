import cv2
import os
import subprocess

video_path = 'video/00000031 (online-video-cutter.com).mp4'
output_dir = 'images-video31/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
def split_video_to_frames(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(frame_count):
        success, frame = cap.read()
        if success:
            cv2.imwrite(os.path.join(output_dir, f'{i:06d}.jpg'), frame)
    cap.release()
    
split_video_to_frames(video_path, output_dir)

subprocess.run(['labelImg', output_dir])
