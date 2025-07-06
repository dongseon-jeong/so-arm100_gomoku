from glob import glob
import shutil
import os
from sklearn.model_selection import train_test_split

# 파일명 수정정
# path = "D:/making/so-arm100_gomoku/vision_model/detection_data/labels/*.txt"
# txt_list = glob(path)
# names = [x.split('/')[-1][16:] for x in txt_list]

# for i in range(len(names)):
#     old_path = txt_list[i]
#     base_path = "D:/making/so-arm100_gomoku/vision_model/detection_data/labels/"
#     new_path = base_path + names[i]

#     shutil.copyfile(old_path, new_path)
#     os.remove(old_path)
# print("comp")

path = "D:\\making\\so-arm100_gomoku\\vision_model\\detection\\detection_data"
img_list = os.listdir(path)
img_list = [os.path.join(path,x) for x in img_list if x.endswith(".jpg")]
seed = 42
# img_list = glob(path)
# print(img_list)

train, val = train_test_split(img_list, test_size = 0.2, random_state = seed)


with open("./train.txt", "w") as f:
    for item in train:
        f.writelines(item+"\n")
with open("./valid.txt", "w") as f:
    for item in val:
        f.writelines(item+"\n")