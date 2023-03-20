import cv2
import os

video_path = 'video/convert.mp4'
output_dir = 'images1/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
j = 0
def split_video_to_frames(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(frame_count):       
            success, frame = cap.read()
            if success:
                if i % 8 == 0 and i > 680:
                    j = int((i - 680)/8) + 1080
                    cv2.imwrite(os.path.join(output_dir, f'{j:06d}.jpg'), frame)
    cap.release()
    
split_video_to_frames(video_path, output_dir)
