from ultralytics import YOLO


def main():
    model = YOLO("yolo11n.pt") 

    results = model.train(
        data="D:/making/so-arm100_gomoku/vision_model/detection/dat.yaml", 
        epochs=25, 
        imgsz=640,
        device=0,
        workers=0  # 🔹 멀티프로세싱 방지
    )


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # 윈도우 환경
    main()