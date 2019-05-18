from closed_eye.eye import *
from closed_eye.utils import *
from customize.main import customization

import sys
import json
import cv2
from imutils.video import VideoStream
import imutils
import dlib

def main(debug=False):
    from pupil_tracker import preprocess, predict, classes

    # 커스터마이제이션 설정이 있는 파일을 열어 ear_thresh 값(eye aspect ratio에 대한 임계값)을 가져옴
    ear_thresh = load_ear_thresh(debug)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        './models/shape_predictor_68_face_landmarks.dat')
    eye_cascade = cv2.CascadeClassifier(
        './models/haarcascade_eye.xml'
    )

    (left_start, left_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (right_start, right_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    vs = VideoStream(src=0, resolution=(1280, 960)).start()
    fileStream = False

    if debug:
        cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Frame', 1000, 800)

    prev_face = None
    prev_idx = 0
    PREV_MAX = 100

    while True:
        if fileStream and not vs.more():
            break

        frame = vs.read()
        frame = imutils.resize(frame, width=960)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        try:
            faces = detector(gray, 0)
            faces = sorted(
                faces,
                key=lambda face: face.width() * face.height(),
                reverse=True)
            # 면적(인식된 범위)이 가장 커다란 사각형(얼굴)을 가져옴
            face = faces[0]

        except IndexError:
            face = None

        if face:
            prev_idx = 0

        if not face:
            if prev_face is not None and prev_idx < PREV_MAX:
                face = prev_face  # 결과가 없는 경우 적절히 오래된(PREV_MAX) 이전 결과를 사용
                prev_idx += 1

        if face:  # 얼굴을 인식한 경우(prev_face를 사용하는 경우 포함)
            prev_face = face  # 저장

            shape = get_shape(predictor, gray, face)

            left_eye_shape = get_eye_shape(shape, left_start, left_end)
            left_ear = get_ear(left_eye_shape)

            right_eye_shape = get_eye_shape(shape, right_start, right_end)
            right_ear = get_ear(right_eye_shape)

            if debug:  # 디버그 모드에서 발견된 결과 표시
                draw_dlib_rect(frame, face)
                # draw_contours(frame, left_eye_shape)
                # draw_contours(frame, right_eye_shape)

            # (x, y, w, h) = face_from_dlib_rect(gray, face)
            # cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0),2)
            face_arr = face_from_dlib_rect(gray, face)
            eyes = eye_cascade.detectMultiScale(face_arr)
            face_height = np.size(face_arr, 0)

            result = -1
            for eye in eyes:
                (x, y, w, h) = eye

                if y + h > face_height * 1 / 2: # 콧구멍 x
                    continue

                # cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0),2)
                eye = face_arr[y:y+h, x:x+w]

                eye = preprocess.apply_threshold(eye)

                percentage, result = predict.prediction(eye)
                result = int(result) # int64 to int

                if debug:
                    print(list(percentage))
                    # pos, label = [
                    #     [(0, 0), 'bottom_left'],
                    #     [(800, 0), 'bottom_right'],
                    #     [(500, 300), 'normal'],
                    #     [(0, 500), 'top_left'],
                    #     [(750, 500), 'top_right']
                    # ][result]
                    pos = (0, 0)
                    label = classes[result]
                    print(label)
                    frame = put_korean(frame, label, pos, fontSacle=30, color='RED')
                break

            print(json.dumps({
                'closed': eye_closed(left_ear, right_ear, ear_thresh, debug),
                'stare': result
            }))

        else:
            print(json.dumps({
                'closed': -1,
                'stare': -1
            }))

        sys.stdout.flush()

        if debug:  # 디버그 모드
            cv2.imshow("Frame", frame)  # 프레임 표시

            # q 키를 눌러 종료
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    cv2.destroyAllWindows()
    vs.stop()


if __name__ == '__main__':
    try:
        debug = int(sys.argv[1])
    except:
        debug = False

    if debug == 2:
        customization()
    else:
        main(debug)
