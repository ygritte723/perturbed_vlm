from torch.utils.data import DataLoader


"""pathFileTrain = '/jet/home/lisun/work/xinliu/images/CheXpert-v1.0-small/train_mod1.csv'
pathFileValid = '/jet/home/lisun/work/xinliu/images/CheXpert-v1.0-small/valid_mod.csv'
pathFileTest = '/jet/home/lisun/work/xinliu/images/CheXpert-v1.0-small/test_mod.csv'

transforms = Compose([Resize(256), CenterCrop(224), ToTensor()])  
# Load dataset
datasetTrain = CheXpertDataSet(pathFileTrain, transforms, policy = "ones")
print("Train data length:", len(datasetTrain))
print(datasetTrain[0][1])
print(datasetTrain[1][1])
print(datasetTrain[2][1])

datasetValid = CheXpertDataSet(pathFileValid, transforms)
print("Valid data length:", len(datasetValid))
print(datasetValid[0][1])
print(datasetValid[1][1])
print(datasetValid[2][1])
datasetTest = CheXpertDataSet(pathFileTest, transforms, policy = "ones")
print("Test data length:", len(datasetTest))
print(datasetTest[0][1])
print(datasetTest[1][1])
print(datasetTest[2][1])
"""
from PIL import Image

im = Image.open(
    "/jet/home/lisun/work/xinliu/images/CheXpert-v1.0-small/train/patient00770/study1/view1_frontal.jpg"
)

im.show()
