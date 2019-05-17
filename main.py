
from pupil_tracker import preprocess, predict
from closed_eye.eye import *
from closed_eye.utils import *

import sys
import json
import cv2
from imutils.video import VideoStream
import imutils
import dlib

def main(debug=False):
    # 커스터마이제이션 설정이 있는 파일을 열어 ear_thresh 값(eye aspect ratio에 대한 임계값)을 가져옴
    ear_thresh = load_ear_thresh(debug)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        './models/shape_predictor_68_face_landmarks.dat')
    eye_cascade = cv2.CascadeClassifier(
        './models/haarcascade_eye.xml'
    )

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

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

            left_eye_shape = get_eye_shape(shape, lStart, lEnd)
            leftEAR = get_ear(left_eye_shape)

            right_eye_shape = get_eye_shape(shape, rStart, rEnd)
            rightEAR = get_ear(right_eye_shape)

            if debug:  # 디버그 모드에서 발견된 결과 표시
                draw_dlib_rect(frame, face)
                draw_contours(frame, left_eye_shape)
                draw_contours(frame, right_eye_shape)

            eyes = eye_cascade.detectMultiScale(face_from_dlib_rect(gray, face))
            # print(eyes)

            for eye in eyes:
                (x, y, w, h) = eye
                eye = gray[y:y+h, x:x+w]

                eye = preprocess.apply_threshold(eye)

                percentage, result = predict.prediction(eye)
                # print(percentage)

                if debug:
                    pos = {
                        'top_left': (0, 0),
                        'top_right': (800, 0),
                        'normal': (800, 300),
                        'bottom_left': (0, 500),
                        'bottom_right': (800, 500)
                    }[result]
                    frame = put_korean(frame, result, pos, fontSacle=30, color='RED')

            print(json.dumps({
                'closed': eye_closed(leftEAR, rightEAR, ear_thresh, debug),
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
    main(debug)
