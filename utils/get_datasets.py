from roboflow import Roboflow

rf = Roboflow(api_key="YW73mUt7roDyutXsAUNs")

# Dataset 1
project_1 = rf.workspace("dataa-byer3").project("thinhlinux-wk7e1")
version_1 = project_1.version(3)
dataset_1 = version_1.download("yolov8-obb")

# Dataset 2
project_2 = rf.workspace("dataa-byer3").project("my-first-project-wucu3")
version_2 = project_2.version(3)
dataset_2 = version_2.download("yolov8-obb")
