import cv2

if __name__ == '__main__':
    # 定数定義
    ESC_KEY = 27     # Escキー
    INTERVAL= 33     # 待ち時間
    FRAME_RATE = 1  # fps

    ORG_WINDOW_NAME = "org"
    GAUSSIAN_WINDOW_NAME = "gaussian"

    DEVICE_ID = 0

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

    # 変換処理ループ
    while end_flag == True:

        # 画像の取得と顔の検出
        img = c_frame
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_list = face_cascade.detectMultiScale(img_gray, minSize=(200, 200))


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
            for (ex, ey, ew, eh) in eyes_list:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), eye_color, thickness = eye_pen_w)



        # フレーム表示
        # cv2.imshow(ORG_WINDOW_NAME, c_frame)
        cv2.imshow(GAUSSIAN_WINDOW_NAME, img)

        # Escキーで終了
        key = cv2.waitKey(INTERVAL)
        if key == ESC_KEY:
            break

        # 次のフレーム読み込み
        end_flag, c_frame = cap.read()


    # 終了処理
    cv2.destroyAllWindows()
    cap.release()