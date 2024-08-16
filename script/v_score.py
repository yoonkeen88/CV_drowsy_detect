from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("runs/detect/train7/weights/best.pt") # �썕�젴 紐⑤뜽 濡쒕뱶

    # Customize validation settings
    validation_results = model.val(data="drowsy detection.v2i.yolov8/data.yaml", imgsz=640, batch=16, conf=0.25, iou=0.6, device="0")
    v_map50_95 = validation_results.box.map  # map50-95
    v_map50 = validation_results.box.map50  # map50
    v_map75 = validation_results.box.map75 # map75
    v_mps = validation_results.box.maps  #媛� 移댄뀒怨좊━�쓽 map50-95 由ъ뒪�듃
    # metrics = model.val(save_json=True)

    print(v_mps, "\n",\
          v_map50, "\n",
          v_map50_95,"\n",
          v_map75,"\n")