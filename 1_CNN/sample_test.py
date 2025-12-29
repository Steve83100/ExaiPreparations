import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as trans
import matplotlib.pyplot as pt
import numpy as np
import vgg11
import vgg13
import vgg16
import resnet



# Model's name is used for model save's name and deciding which model is actually used.
MODEL_NAME = "vgg13"
# MODEL_NAME = "resnet"

# Dataset's name is used in model save's name and deciding which dataset is used.
DATASET_NAME = "cifar-10"
CLASSES_NUM = 10

# DATASET_NAME = "tiny-imagenet-200"
# CLASSES_NUM = 200

BATCH_SIZE = 8

DATASET_PATH = "~/syang/Datasets/"
# Needs to be run in the directory containing "ExaiPreparations/" !!!
ROOT_PATH = "./ExaiPreparations/" # Path of ExaiPreparations
MODEL_PATH = ROOT_PATH + "ModelSaves/"
OUTPUT_PATH = ROOT_PATH + "Outputs/"

# if torch.cuda.is_available(): # Use CUDA if available
#     DEVICE = torch.device("cuda:4")
#     print("Using device: " + torch.cuda.get_device_name(DEVICE))
# else:
#     DEVICE = torch.device("cpu")
#     print("Using device: CPU")
    
DEVICE = torch.device("cpu") # just use CPU for now



# unnormalize and show images
def imshow(img, mean, std):
    m = torch.tensor(mean).view(-1,1,1)
    s = torch.tensor(std).view(-1,1,1)
    # print(img.size())
    # print(m.size())
    # print(s.size())
    img = torch.clip(img.mul_(s).add_(m), 0, 1)
    npimg = torchvision.utils.make_grid(img).numpy()
    pt.imshow(np.transpose(npimg, (1, 2, 0)))
    pt.savefig(OUTPUT_PATH + "samples-" + DATASET_NAME + '.png')
    # pt.show()



if __name__ == '__main__':
    
    
    
    # Retrive and transform data
    print("Fetching dataset: " + DATASET_NAME)
    match DATASET_NAME:
        case "cifar-10":
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2023, 0.1994, 0.2010]
            train_transform = trans.Compose([
                trans.RandomCrop(32, padding=4),
                trans.RandomHorizontalFlip(),
                trans.ToTensor(),
                trans.Normalize(mean, std)
                ])
            train_set = torchvision.datasets.CIFAR10(root=DATASET_PATH, train=True, download=True, transform=train_transform)
            train_loader = data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
            
            test_transform = trans.Compose([
                trans.ToTensor(),
                trans.Normalize(mean, std)
                ])
            test_set = torchvision.datasets.CIFAR10(root=DATASET_PATH, train=False, download=True, transform=test_transform)
            test_loader = data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
            classes = ("plane","car","bird","cat","deer","dog","frog","horse","ship","truck")
            
        # case "tiny-imagenet-200":
        #     train_transform = trans.Compose([
        #         trans.RandomCrop(64, padding=4),
        #         trans.RandomHorizontalFlip(),
        #         trans.ToTensor(),
        #         trans.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        #         ])
        #     train_set = torchvision.datasets.ImageFolder(root=DATASET_PATH+"tiny-imagenet-200/train", transform=train_transform)
        #     train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
            
        #     test_transform = trans.Compose([
        #         trans.ToTensor(),
        #         trans.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        #         ])
        #     test_set = torchvision.datasets.ImageFolder(root=DATASET_PATH+"tiny-imagenet-200/val", transform=test_transform)
        #     test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
            
        case _:
            raise Exception("Unrecognized dataset: " + DATASET_NAME)

    
    
    # Instantiate model
    print("Creating model: " + MODEL_NAME)
    match MODEL_NAME:
        case "vgg11":
            model = vgg11.Net(CLASSES_NUM)
        case "vgg13":
            model = vgg13.Net(CLASSES_NUM)
        case "vgg16":
            model = vgg16.Net(CLASSES_NUM)
        case "resnet":
            model = resnet.Net(CLASSES_NUM)
        case _:
            raise Exception("Unrecognized model: " + MODEL_NAME)
    model.to(DEVICE)

    # Load model from disk
    print("Loading model from disk...")
    path = MODEL_PATH + MODEL_NAME + '-' + DATASET_NAME + '.pth'
    model.load_state_dict(torch.load(path, weights_only=True))
    


    # get some test images
    print("Generating test samples...")
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    # print(images.size())

    # show images
    imshow(images, mean, std)
    
    # print actual labels
    print("Real: " + ' '.join(f'{classes[labels[j]]:5s}' for j in range(BATCH_SIZE)))
    
    # make prediction
    images.to(DEVICE)
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    print("Pred: " + ' '.join(f'{classes[predicted[j]]:5s}' for j in range(BATCH_SIZE)))