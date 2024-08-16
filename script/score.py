from ultralytics import YOLO

if __name__ == "__main__":
    # 모델 로드
    model = YOLO("runs/detect/train7/weights/best.pt")  # 사용자 정의 모델 로드

    # 모델 검증
    metrics = model.val(data="drowsy detection.v2i.yolov8/data.yaml")  # 데이터셋 및 설정 사용
    
    map50_95 = metrics.box.map  # map50-95
    map50 = metrics.box.map50  # map50
    map75 = metrics.box.map75 # map75
    mps = metrics.box.maps  #각 카테고리의 map50-95 리스트

    precisions = metrics.box.precision  # Precision
    recalls = metrics.box.recall  # Recall

    # True Positives, False Positives, False Negatives 가져오기
    tp = metrics.box.tp.sum()
    fp = metrics.box.fp.sum()
    fn = metrics.box.fn.sum()

    # Accuracy 계산
    accuracy = tp / (tp + fp + fn)
    print(f"Accuracy: {accuracy:.4f}")

    # F1score = 2*(precisions*recalls)/(precisions+recalls)
   
    # metrics = model.val(save_json=True)

    # print(mps, "\n", map50, "\n", map50_95,"\n", map75,"\n", precisions,"\n", recalls)