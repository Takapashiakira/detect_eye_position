import cv2
from datetime import datetime
from time import time
from time import sleep
import numpy as np


if __name__ == '__main__':
    # 定数定義
    ESC_KEY = 27     # Escキー
    INTERVAL= 33     # 待ち時間
    FRAME_RATE = 1  # fps

    ORG_WINDOW_NAME = "org"
    GAUSSIAN_WINDOW_NAME = "gaussian"

    DEVICE_ID = 0

    today = datetime.now().strftime("%s")
    eye_position_fn = today + ".txt"

    # 分類器の指定
    face_cascade_file = "haarcascades/haarcascade_frontalface_alt2.xml"
    face_cascade = cv2.CascadeClassifier(face_cascade_file)

    eye_cascade_file = "haarcascades/haarcascade_eye.xml"
    eye_cascade = cv2.CascadeClassifier(eye_cascade_file)

    # カメラ映像取得
    cap = cv2.VideoCapture(DEVICE_ID)

    # 初期フレームの読込
    end_flag, c_frame = cap.read()
    height, width, channels = c_frame.shape

    # ウィンドウの準備
    # cv2.namedWindow(ORG_WINDOW_NAME)
    cv2.namedWindow(GAUSSIAN_WINDOW_NAME)


    past_eye_list = []
    past_face_list =[[510, 232, 346, 346]]


    # 変換処理ループ
    while end_flag == True:

        try:
            # 画像の取得と顔の検出
            img = c_frame
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face_list = face_cascade.detectMultiScale(img_gray, minSize=(200, 200))

            face_list = past_face_list if len(face_list)>1 else face_list


            # 検出した顔に印を付ける
            for (x, y, w, h) in face_list:
                face_color = (0, 0, 225)
                face_pen_w = 10
                cv2.rectangle(img, (x, y), (x+w, y+h), face_color, thickness = face_pen_w)

                roi_gray = img_gray[y:y+int(h/2), x:x+w]
                roi_color = img[y:y+int(h/2), x:x+w]

                eyes_list = eye_cascade.detectMultiScale(roi_gray, minSize=(40, 40), minNeighbors=5)
                eye_color = (0, 255, 0)
                eye_pen_w = 10

                if len(eyes_list) == 2:
                    if abs(eyes_list[0][0] - eyes_list[1][0]) > w/3:
                        past_eye_list = eyes_list
                    else:
                        eyes_list = past_eye_list
                else:
                    eyes_list = past_eye_list
            
            # 検出した目に印を付ける
                eyes_coodinates_list = []
                for (ex, ey, ew, eh) in eyes_list:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), eye_color, thickness = eye_pen_w)
                    ex = ex + (ew/2) + x
                    ey = ey + (eh/2) + y
                    eyes_coodinates_list.append([ex]+[ey])

            # 両目の中点の座標を求める
            eye_middle_position = []
            middle_x = (eyes_coodinates_list[0][0] + eyes_coodinates_list[1][0])/2
            middle_y = (eyes_coodinates_list[0][1] + eyes_coodinates_list[1][1])/2

            eye_middle_position.append(middle_x)
            eye_middle_position.append(middle_y)


            # 両目の距離を求める
            a = np.array(eyes_coodinates_list[0])
            b = np.array(eyes_coodinates_list[1])
            d = np.linalg.norm(a-b)

            # # print(eye_middle_position)
            # # print(right_eye_list[0])
            
            # result_str = ",".join(eye_middle_position)
            # print(eye_middle_position)
            # 
            
            timestamp = str(datetime.now().timestamp())
            eye_middle_position.append(d)
            eye_middle_position.append(timestamp)

            eye_middle_position_str = [str(n) for n in eye_middle_position]
            result_str = ",".join(eye_middle_position_str)
            print(result_str)

            # ファイルに書き込み
            with open(eye_position_fn, mode="a") as f_b:
                f_b.writelines(result_str + "\n")


            # フレーム表示
            # cv2.imshow(ORG_WINDOW_NAME, c_frame)
            cv2.imshow(GAUSSIAN_WINDOW_NAME, img)

            # Escキーで終了
            key = cv2.waitKey(INTERVAL)
            if key == ESC_KEY:
                break

            # 次のフレーム読み込み
            end_flag, c_frame = cap.read()
            sleep(0.05)
        
        except:
            pass


    # 終了処理
    cv2.destroyAllWindows()
    cap.release()