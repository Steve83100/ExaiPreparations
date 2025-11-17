import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torchvision
import torchvision.transforms as trans
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as pt
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
# CLASSES_NUM = 200 # Number of classes the dataset has

# Hyperparameters
EPOCHS = 50
BATCH_SIZE = 64
BATCH_GROUP_NUM = 100 # Number of batches in a group. Will record progress after each trained group
START_LR = 0.01

DATASET_PATH = "~/syang/Datasets/"
MODEL_PATH = "./ModelSaves/"
OUTPUT_PATH = "./Outputs/"

DEVICE = torch.device("cuda:5" if torch.cuda.is_available() else "cpu") # Use CUDA if available
print("Using device: " + torch.cuda.get_device_name(DEVICE))



if __name__ == '__main__':
    
    # Retrive and transform data
    match DATASET_NAME:
        case "cifar-10":
            train_transform = trans.Compose([
                trans.RandomCrop(32, padding=4),
                trans.RandomHorizontalFlip(),
                trans.ToTensor(),
                trans.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
            train_set = torchvision.datasets.CIFAR10(root=DATASET_PATH, train=True, download=True, transform=train_transform)
            train_loader = data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
            
            test_transform = trans.Compose([
                trans.ToTensor(),
                trans.RandomCrop(32, padding=4),
                trans.RandomHorizontalFlip(),
                trans.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
            test_set = torchvision.datasets.CIFAR10(root=DATASET_PATH, train=False, download=True, transform=test_transform)
            test_loader = data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
            # classes = ("plane","car","bird","cat","deer","dog","frog","horse","ship","truck")
            
        case "tiny-imagenet-200":
            transform = trans.Compose([trans.ToTensor(), trans.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            train_set = torchvision.datasets.ImageFolder(root=DATASET_PATH+"tiny-imagenet-200/train", transform=transform)
            test_set = torchvision.datasets.ImageFolder(root=DATASET_PATH+"tiny-imagenet-200/val", transform=transform)
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
        case _:
            raise Exception("Unrecognized dataset: " + DATASET_NAME)
    
    # for i, (items, labels) in enumerate(train_loader, 0):
    #     if i==0 :
    #         print(items.size())
    #         print(items[0])
    
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
    # print(model)


    # Create criterion and optimizer
    criterion = nn.CrossEntropyLoss()
        # The loss calculator does 2 things for us:

        # First, apply a final softmax layer to model's output, so that yPred becomes a probability vector of classes,
        # such as yPred_n = [1, 2, 3] becomes yPred_n = [0.09, ]

        # Next, automatically detect if labels are given in the form of class indexes or probability vectors.
        # If labels are class indexes (y_n = 0, 1, ..., C-1), then use one-hot encoding to transform into probability vectors,
        # such as y_n = 1 becomes y_n = [0, 1, 0, ..., 0].

        # After the above processing, the actual log-loss is calculated: l_n = - sum(y_n_c * log(yPred_n_c))

    optimizer = optim.SGD(model.parameters(), lr=START_LR, momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS/10)


    # Train model and record training process
    print("----------------Training begins----------------")
    
    epochs = []
    
    train_losses = []
    train_accuracies = []
    
    valid_losses = []
    valid_accuracies = []

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}:")
        total = 0
        correct = 0
        running_loss = 0.0
        
        for i, (items, labels) in enumerate(train_loader, 0):
            items, labels = items.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            
            outputs = model(items)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() # Accumulate loss
            _, indexes = torch.max(outputs.data, dim=1) # Manually retrive predicted class index from outputs
            correct += (indexes == labels).sum().item() # Count the number of correct predictions
            total += labels.size(0)
            
            if i % BATCH_GROUP_NUM == (BATCH_GROUP_NUM - 1): # Notify training progress after [BATCH_GROUP_NUM] batches
                print(f'[Batches trained: {i + 1}] ')
                
        epochs.append(epoch)
        
        print(f'Average training loss: {running_loss/total:.6f}, accuracy: {correct/total * 100:.3f}% on {total:.0f} samples')
        train_losses.append(running_loss/total)
        train_accuracies.append(correct/total * 100)
        
        print("Validating...")
        vtotal = 0
        vcorrect = 0
        vloss = 0.0
        with torch.no_grad():
            for (items, labels) in test_loader:
                items, labels = items.to(DEVICE), labels.to(DEVICE)
                outputs = model(items)
                vloss += (criterion(outputs, labels)).item()
                _, indexes = torch.max(outputs.data, dim=1)
                vcorrect += (indexes == labels).sum().item()
                vtotal += labels.size(0)
                accuracy = vcorrect / vtotal * 100
        print(f'Average validation Loss: {vloss/vtotal:.6f}, Accuracy: {accuracy:.3f}% on {vtotal:.0f} samples')
        valid_losses.append(vloss/vtotal)
        valid_accuracies.append(accuracy)
        
        # scheduler.step() # Update learning rate after an epoch

    print("----------------Training completed----------------")
    print("Saving model to disk...")


    # Save model to disk
    path = MODEL_PATH + MODEL_NAME + '-' + DATASET_NAME + '.pth'
    torch.save(model.state_dict(), path)


    # Load model from disk (just for demonstration)
    # print("Loading model from disk...")
    # model = resnet.Net(CLASSES_NUM)
    # model.load_state_dict(torch.load(path, weights_only=True))
    
    
    # Plot training and validating process
    print("Plotting loss and accuracy graph...")
    fig = pt.figure()
    
    loss_graph = fig.add_subplot(121)
    loss_graph.plot(epochs, train_losses, label = "Training Loss")
    loss_graph.plot(epochs, valid_losses, label = "Validation Loss")
    loss_graph.set_title("Loss")
    loss_graph.set_xlabel("Epoch")
    loss_graph.legend()
    
    accuracy_graph = fig.add_subplot(122)
    accuracy_graph.plot(epochs, train_accuracies, label = "Training Accuracy")
    accuracy_graph.plot(epochs, valid_accuracies, label = "Validation Accuracy")
    accuracy_graph.set_title("Accuracy")
    accuracy_graph.set_xlabel("Epoch")
    accuracy_graph.legend()
    
    pt.savefig(OUTPUT_PATH + MODEL_NAME + '-' + DATASET_NAME + '.png')
    # pt.show()
