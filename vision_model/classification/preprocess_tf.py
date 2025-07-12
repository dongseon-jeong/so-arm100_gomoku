from glob import glob
import cv2
import os
from ultralytics import YOLO

# 원본 이미지 불러오기
fols = ["classification_data","classification_data2"]

for fol in fols:
    path = "D:\\making\\so-arm100_gomoku\\vision_model\\classification\\"+fol
    img_list = os.listdir(path)


    model = YOLO("D:\\making\\so-arm100_gomoku\\vision_model\\detection\\runs\\detect\\train9\\weights\\best.pt")

    i = 0

    for i in range(len(img_list)):

        img = img_list[i]

        image_path = path+"\\"+img
        result = model.predict(image_path)


        image = cv2.imread(image_path)
        detections = result[0].boxes.xyxy.cpu().numpy().tolist()[:3]
        print(detections)
        # 저장 폴더 만들기
        save_dir = 'crop_image\\' + fol
        os.makedirs(save_dir, exist_ok=True)

        # 디텍션된 박스 크롭 후 저장
        x1= int(detections[0][0])
        x2= int(detections[0][2])
        y1= int(detections[0][1])
        y2= int(detections[0][3])
        cropped = image[y1:y2, x1:x2]
        save_path = os.path.join("D:\\making\\so-arm100_gomoku\\vision_model\\classification",save_dir,img)
        cv2.imwrite(save_path, cropped)

        print("크롭 완료")
