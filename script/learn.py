from ultralytics import YOLO
import torch

def main():
    # CUDA �뵒諛붿씠�뒪 �솗�씤
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # YOLOv10 紐⑤뜽 濡쒕뱶 諛� CUDA濡� �씠�룞
    model = YOLO('C:/python_project/CV_drowsy_detect/defalt_weight/yolov10n.pt')

    # �븰�뒿 �꽕�젙
    EPOCHS = 100
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    F_LEARNING_RATE = 0.001
    IMAGE_SIZE = 640
    OPTIMIZER = "AdamW"

    # �븰�뒿 �떎�뻾
    model.train(data = 'C:/python_project/CV_drowsy_detect/drowsy detection.v2i.yolov8/data.yaml', 
                epochs=EPOCHS, 
                batch=BATCH_SIZE, 
                imgsz=IMAGE_SIZE, 
                lr0=LEARNING_RATE, 
                lrf=F_LEARNING_RATE, 
                optimizer=OPTIMIZER)

if __name__ == '__main__':
    main()
