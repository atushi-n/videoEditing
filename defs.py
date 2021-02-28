import cv2
import dlib
import face_recognition
import numpy as np
from PIL import Image, ImageFont, ImageDraw

def mosaic_area(src, x, y, width, height, ratio=0.1):#モザイクをエリア指定してかけます
    dst = src.copy()
    dst[y:y + height, x:x + width] = mosaic(dst[y:y + height, x:x + width], ratio)
    return dst

def mosaic(src, ratio=0.2):#モザイクを全体にかけます
    small = cv2.resize(src, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
    return cv2.resize(small, src.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

def oneshot(cap_file):
    res, frame = cap_file.read()
    print(type(frame))


    print(frame.shape)
    face_locs = face_recognition.face_locations(frame, model="cnn") #上、右、下、左



    print(face_locs)

    face_locs = face_locs[0]

    top = face_locs[0]
    right = face_locs[1]
    bottom = face_locs[2]
    left= face_locs[3]

    #顔に四角を描画
    #frame = cv2.rectangle(frame, (top,right), (bottom, left), (255, 255, 255),-1)  # 座標　カラー

    #face = frame[top:bottom, left:right]  # 高さS 高さ, 横幅S 横幅
    #face = mosaic(face)
    frame = mosaic_area(frame,left, top, right-left, bottom-top)
    cv2.imshow('frame', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('output_image.jpg', frame)







