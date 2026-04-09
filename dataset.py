import os
import kagglehub

os.environ["KAGGLEHUB_CACHE"] = "./"

# 2. 像往常一样下载
path = kagglehub.dataset_download("matthewjansen/ucf101-action-recognition")

print("数据集已下载至:", path)


