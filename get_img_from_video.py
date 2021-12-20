import cv2
import os

video_path = f"{os.getcwd()}/data/video/room1"

image_save_path = f"{os.getcwd()}/data/image/room1"

#get video from video path
for video in os.listdir(video_path):

   cap = cv2.VideoCapture(f"{video_path}/{video}")

   idx = 1

   while cap.isOpened():

      flag,frame = cap.read()

      if flag == False:

         break

      cv2.imwrite(f"{image_save_path}/{video[:-4]}_{idx}.jpg",frame)

      idx = idx + 1


