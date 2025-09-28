import glob
import os
from shutil import move
from os import rmdir

# Original validation folder has structure:
#
# val
# |-- val_annotations.txt (A file specifying the type of each image, such as "val_0.JPEG" belongs to type "n03444034")
# |-- images
#     |-- val_0.JPEG
#     |-- val_1.JPEG
#     |-- ...
#
# This piece of code transforms it into structure:
#
# val
# |-- n01443537 (The class type. All images in this folder have type "n01443537")
#     |-- n01443537_0.JPEG
#     |-- n01443537_1.JPEG
#     |-- ...
# |-- n01629819
#     |-- n01629819_0.JPEG
#     |-- n01629819_1.JPEG
#     |-- ...
# |-- ...
#
# This structure can be loaded with torchvision.datasets.ImageFolder.
# ImageFolder() will automatically scan through the structure, assign indexes to class types, and create (image, type) tuples

target_folder = 'D:/Coding/DeepLearning/Datasets/tiny-imagenet-200/val/'

val_dict = {}
with open('D:/Coding/DeepLearning/Datasets/tiny-imagenet-200/val/val_annotations.txt', 'r') as f:
    for line in f.readlines():
        split_line = line.split('\t')
        val_dict[split_line[0]] = split_line[1]
        
paths = glob.glob('D:/Coding/DeepLearning/Datasets/tiny-imagenet-200/val/images/*')
for path in paths:
    # print(path)
    file = path.split('\\')[-1]
    folder = val_dict[file]
    if not os.path.exists(target_folder + str(folder)):
        os.mkdir(target_folder + str(folder))
    dest = target_folder + str(folder) + '/' + str(file)
    move(path, dest)
    
rmdir('D:/Coding/DeepLearning/Datasets/tiny-imagenet-200/val/images')



# Additionally, original training folder has structure:
#
# train
# |-- n01443537
#     |-- images
#         |-- n01443537_0.JPEG
#         |-- n01443537_1.JPEG
#         |-- ...
# |-- n01629819
#     |-- images
#         |-- n01629819_0.JPEG
#         |-- n01629819_1.JPEG
#         |-- ...
# |-- ...
#
# This piece of code deletes the annoying "images" directory:
#
# train
# |-- n01443537
#     |-- n01443537_0.JPEG
#     |-- n01443537_1.JPEG
#     |-- ...
# |-- n01629819
#     |-- n01629819_0.JPEG
#     |-- n01629819_1.JPEG
#     |-- ...
# |-- ...
#
# Now this structure can be loaded too.

paths = glob.glob('D:/Coding/DeepLearning/Datasets/tiny-imagenet-200/train/*')
for path in paths:
    # print(path)
    imagesPath = glob.glob(path + "/images/*")
    for image in imagesPath:
        # print(image)
        move(image, path)
    rmdir(path + "/images")