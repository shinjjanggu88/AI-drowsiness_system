
import cv2
import mediapipe as mp
import math
import time
import winsound
import threading

# 얼굴 인식 준비
mp_face_mesh = mp.solutions.face_mesh

# 눈 주변 특징점 번호
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# 졸음 감지 기준
EYE_CLOSED_RATIO = 0.23
CLOSED_SECONDS = 1.5

# 사람마다 눈 감은 시간과 알람 상태를 따로 저장
closed_start = {}
alarm_playing = {}


# 눈의 세로/가로 비율 계산
def eye_ratio(eye_points):
    vertical1 = math.dist(eye_points[1], eye_points[5])
    vertical2 = math.dist(eye_points[2], eye_points[4])
    horizontal = math.dist(eye_points[0], eye_points[3])

    if horizontal == 0:
        return 0

    return (vertical1 + vertical2) / (2.0 * horizontal)


# 카메라 설정
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

with mp_face_mesh.FaceMesh(
    max_num_faces=5,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while True:
        ret, frame = cap.read()

        if not ret:
            print("카메라 화면을 가져올 수 없습니다.")
            break

        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            faces = results.multi_face_landmarks
            active_indices = set()

            # 감지된 얼굴을 하나씩 처리
            for face_index, face in enumerate(faces):
                active_indices.add(face_index)

                def get_eye_points(indices):
                    return [
                        (
                            int(face.landmark[i].x * width),
                            int(face.landmark[i].y * height)
                        )
                        for i in indices
                    ]

                left_points = get_eye_points(LEFT_EYE)
                right_points = get_eye_points(RIGHT_EYE)

                left_ratio = eye_ratio(left_points)
                right_ratio = eye_ratio(right_points)
                ratio = (left_ratio + right_ratio) / 2

                # 눈 주변 특징점 표시
                for x, y in left_points + right_points:
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                # 처음 감지된 얼굴의 상태 초기화
                if face_index not in closed_start:
                    closed_start[face_index] = None
                    alarm_playing[face_index] = False

                # 눈을 감았는지 확인
                if ratio < EYE_CLOSED_RATIO:
                    if closed_start[face_index] is None:
                        closed_start[face_index] = time.time()

                    elapsed = time.time() - closed_start[face_index]

                    if elapsed >= CLOSED_SECONDS:
                        # 같은 눈 감김 상태에서는 알람 한 번만 실행
                        if not alarm_playing[face_index]:
                            alarm_playing[face_index] = True

                            threading.Thread(
                                target=winsound.Beep,
                                args=(1000, 500),
                                daemon=True
                            ).start()

                        status = "DROWSINESS WARNING!"
                        color = (0, 0, 255)

                    else:
                        status = "Eyes closed..."
                        color = (0, 255, 255)

                else:
                    # 눈을 뜨면 해당 얼굴의 상태 초기화
                    closed_start[face_index] = None
                    alarm_playing[face_index] = False

                    status = "Eyes open"
                    color = (0, 255, 0)

                # 얼굴마다 상태 표시
                x, y = left_points[0]
                cv2.putText(
                    frame,
                    f"Person {face_index + 1}: {status}",
                    (20, 40 + face_index * 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color,
                    2
                )

            # 사라진 얼굴의 상태 정리
            for index in list(closed_start.keys()):
                if index not in active_indices:
                    closed_start.pop(index, None)
                    alarm_playing.pop(index, None)

            # 현재 감지된 인원 표시
            cv2.putText(
                frame,
                f"People detected: {len(faces)}",
                (20, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

        else:
            # 얼굴이 없으면 상태 초기화
            closed_start.clear()
            alarm_playing.clear()

            cv2.putText(
                frame,
                "Face not detected",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        cv2.imshow("AI Drowsiness Detection", frame)

        # q를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
