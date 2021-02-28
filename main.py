import cv2
import dlib
import face_recognition
import numpy as np
from PIL import Image
import defs
from defs import oneshot

print(dlib.DLIB_USE_CUDA)  # GPUが有効になっているか
out_name = "ero_out.mp4"
movie_name = "ero.mp4"
cap_file = cv2.VideoCapture(movie_name)







#動画を開く
if cap_file.isOpened():

    #動画の情報を表示
    print('frame width:', cap_file.get(cv2.CAP_PROP_FRAME_WIDTH))
    print('frame height:', cap_file.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print('FPS:', cap_file.get(cv2.CAP_PROP_FPS))
    print('frame count:', cap_file.get(cv2.CAP_PROP_FRAME_COUNT))
    print('video play time:', int(cap_file.get(cv2.CAP_PROP_FRAME_COUNT) / cap_file.get(cv2.CAP_PROP_FPS)))

    #入力待ち０ならワンショット１ならオール
    choice = input("->")

    if choice == '0':
        oneshot(cap_file)

    if choice == '1':
        w = int(cap_file.get(cv2.CAP_PROP_FRAME_WIDTH))  # width
        h = int(cap_file.get(cv2.CAP_PROP_FRAME_HEIGHT))  # height
        fps = int(cap_file.get(cv2.CAP_PROP_FPS))  # fps
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # MP4フォーマット
        video = cv2.VideoWriter(out_name, fourcc, fps, (w, h))  # 書き込み先のパス,　フォーマット, fps

        delay = 1

        while True:
            res, frame = cap_file.read()

            if res:  # trueでなければbreakする
                frame = Image.fromarray(frame)

                frame = np.array(frame)

                face_locs = face_recognition.face_locations(frame, model="cnn", number_of_times_to_upsample=1)  # 上、右、下、左
                if len(face_locs) != 0:
                    print(face_locs)
                    face_locs = face_locs[0]
                    top = face_locs[0]
                    right = face_locs[1]
                    bottom = face_locs[2]
                    left = face_locs[3]
                    frame = defs.mosaic_area(frame, left, top, right - left, bottom - top, ratio=0.07)
                else:
                    print("このフレームから顔が見つからなかった")

                cv2.imshow('frame', frame)
                video.write(frame)
                if cv2.waitKey(delay) & 0xFF == ord('q'):
                    break

            else:
                #				cap_file.set(cv2.CAP_PROP_POS_FRAMES, 0)
                break
        print("完了　０を押してください")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        video.release()

    print("終了")
    cap_file.release()  # 閉じる
else:
    print("オープンに失敗")
