import clip
import glob
import os
import random
import shutil
import time
import torch
import torchvision
import torchvision.transforms as trans
from torch.utils.data import Dataset, random_split, DataLoader



# ====================================================================

random.seed(86)
torch.manual_seed(86)
DATASET_PATH = "D:/Coding/DeepLearning/Datasets/"



# ====================================================================

# Filter dataset down to 50 classes (only needed to be run once)
# After filtering, we directly delete extra classes from our new dataset folder which was a copy of tinyImagenet,
# instead of taking Subset during data preprocessing, because filtering is time consuming (>40s),
# and we don't want to do it every time we load the dataset.

# if __name__ == "__main__":
#     dataset = torchvision.datasets.ImageFolder(root=DATASET_PATH+"tiny-imagenet-200/train/")
#     all_classes = list(dataset.class_to_idx.keys())
#     selected_classes = random.sample(all_classes, 50)
#     print(selected_classes)
    
#     curr = time.time()
    
#     folder = DATASET_PATH+'micro-imagenet-50/'
#     paths = glob.glob(folder + '*')
#     for path in paths:
#         class_code = path.split('\\')[-1]
#         # print(class_code)
#         if class_code not in selected_classes:
#             shutil.rmtree(folder + str(class_code))
            
#     print(time.time() - curr)



# ====================================================================

# Calculate mean and std (only needed to be run once)
# Again, we calculate mean and std beforehands since the calculation process is time consuming
# Note that this might not be necessary, as CLIP has its own normalizing parameters in "preprocess"

# if __name__ == "__main__":
#     train_set = torchvision.datasets.ImageFolder(root=DATASET_PATH+"micro-imagenet-50/", transform=trans.ToTensor())
#     print(len(train_set))
#     loader = DataLoader(train_set, batch_size=64, shuffle=False, num_workers=4)

#     mean = 0.
#     std = 0.
#     total_images = 0
    
#     curr = time.time()

#     for images, _ in loader:
#         batch_samples = images.size(0) # batch size
#         images = images.view(batch_samples, images.size(1), -1) # reshape to (batch, channels, H*W)
#         mean += images.mean(2).sum(0) # sum mean across all images
#         std += images.std(2).sum(0) # sum std across all images
#         total_images += batch_samples
            
#     print(time.time() - curr)

#     mean /= total_images
#     std /= total_images

#     print(mean) # [0.4829, 0.4499, 0.4026]
#     print(std) # [0.2285, 0.2248, 0.2253]



# ====================================================================

# TODO: Split dataset and apply different transforms with my own implementation of ImageFolder

class TinyImageNet(Dataset):
    """
    TinyImageNet dataset with 50 classes and 25000 images, created from filtering tiny-imagenet-200.
    """
    def __init__(self, additional_preprocess):
        super().__init__()
        
        self.train = True # Dataset mode, decides 
        
        # Define preprocessing methods
        print("Processing data...")
        self.train_transform = trans.Compose([ # No normalization since CLIP has its own
            trans.RandomHorizontalFlip(),
            trans.RandomVerticalFlip(),
            additional_preprocess
            ])
        self.test_transform = additional_preprocess
        
        self.dataset = torchvision.datasets.ImageFolder(root=DATASET_PATH+"micro-imagenet-50/")
        print("Dataset size:", len(self.dataset))
        
        # Map class code to its name, such as "n03908204" to "pencil"
        self.code_name_dict = {}
        with open(DATASET_PATH+"tiny-imagenet-200/words.txt", 'r') as f:
            for line in f.readlines():
                split_line = line.split('\t')
                self.code_name_dict[split_line[0]] = split_line[1].strip()
        f.close()

    def __getitem__(self, index):
        image, label_id = self.dataset[index]
        return image, label_id
    
    def __len__(self):
        return len(self.dataset)



def get_data(batch_size, train_valid_test_ratio, additional_preprocess):
    """
    Get train, validate and test loaders of filtered tiny-imagenet-200.
        
        :param train_valid_test_ratio: List of int or float such as [0.8, 0.1, 0.1]
    
    """
    dataset = TinyImageNet(additional_preprocess)
    train_set, valid_set, test_set = random_split(dataset, train_valid_test_ratio)
    train_loader = DataLoader(train_set, shuffle = True, batch_size = batch_size)
    valid_loader = DataLoader(valid_set, shuffle = False, batch_size = batch_size)
    test_loader = DataLoader(test_set, shuffle = False, batch_size = batch_size)
    
    return dataset, train_loader, valid_loader, test_loader



# ====================================================================

# Test code

if __name__ == "__main__":
    model, preprocess = clip.load("ViT-B/32")
    
    dataset = TinyImageNet(preprocess)
    image, label_id = dataset[499]
    print(image.shape)
    print(label_id)
    print(dataset.code_name_dict[dataset.dataset.classes[label_id]])
    
    image, label_id = dataset[500]
    print(label_id)
    print(dataset.code_name_dict[dataset.dataset.classes[label_id]])
    
    dataset, train_loader, valid_loader, test_loader = get_data(12, [0.8, 0.1, 0.1], preprocess)
    images, label_ids = next(iter(train_loader))
    print(images.shape)
    print(label_ids)