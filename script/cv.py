import cv2
from ultralytics import YOLO
import torch

# 모델 가중치 파일 경로 정의
weights_path = '/Users/angwang-yun/Desktop/Project/yolov10_n_learning/runs/customed_weights/best.pt'
model_weight_path = '/Users/angwang-yun/Desktop/Project/YOLOv10_n_weight_epoch100/weights/yolov10n.pt'

# YOLOv10 모델 로드
model = YOLO(model_weight_path)

# 사용자 정의 가중치를 모델에 로드
model.load_state_dict(torch.load(weights_path))

def run_webcam_detection(model):
    cap = cv2.VideoCapture(0)  # 웹캠 열기; '0'은 기본 카메라 장치 인덱스

    if not cap.isOpened():
        print("오류: 웹캠을 열 수 없습니다.")
        return

    print("탐지를 종료하려면 'q'를 누르세요.")

    while True:
        # 프레임을 하나씩 캡처
        ret, frame = cap.read()

        if not ret:
            print("오류: 프레임을 캡처할 수 없습니다.")
            break

        # YOLOv10 모델을 사용하여 객체 탐지 수행
        results = model(frame)

        # 탐지 결과를 프레임에 주석으로 추가
        annotated_frame = results[0].plot()

        # 주석이 추가된 프레임을 화면에 표시
        cv2.imshow("REAL-TIME DETECTION", annotated_frame)

        # 'q' 키를 눌러서 종료할 수 있음
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 웹캠 자원 해제 및 모든 OpenCV 창 닫기
    cap.release()
    cv2.destroyAllWindows()

# 웹캠 탐지 실행
run_webcam_detection(model)
