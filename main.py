import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torchvision
import torchvision.transforms as trans
import matplotlib.pyplot as pt
import vgg
import resnet


# Model's name is used for model save's name. But CANNOT decide which model is actually used.
MODEL_NAME = "resnet"

# Dataset's name is used in model save's name AND deciding which dataset is used.
# DATASET_NAME = "cifar-10"
DATASET_NAME = "tiny-imagenet-200"
CLASSES_NUM = 200 # Number of classes the dataset has

# Training hyperparameters. Not including optimizer hyperparameters (lr, momentum)
EPOCHS = 4
BATCH_SIZE = 10
BATCH_GROUP_NUM = 200 # Number of batches in a group. Will record progress after each trained group



def validate(model, test_loader, criterion, device):
    print("Validating...")
    total = 0.0
    correct = 0.0
    with torch.no_grad():
        for (items, labels) in test_loader:
            items, labels = items.to(device), labels.to(device)
            outputs = model(items)
            loss = criterion(outputs, labels)
            _, indexes = torch.max(outputs.data, dim=1)
            correct += (indexes == labels).sum().item()
            total += labels.size(0)
            accuracy = correct / total * 100
    print(f'Validation Loss: {loss:.3f}, Accuracy: {accuracy:.3f}%')
    return loss, accuracy



if __name__ == '__main__':
    
    # Retrive and transform data
    print("Fetching data...")
    match DATASET_NAME:
        case "cifar-10":
            print("Using dataset cifar-10")
            transform = trans.Compose([trans.ToTensor(), trans.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # Scale [0,1] to [-1,1]
            train_set = torchvision.datasets.CIFAR10(root="D:/Coding/DeepLearning/Datasets",
                                                     train=True, download=True, transform=transform)
            train_loader = data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
            test_set = torchvision.datasets.CIFAR10(root="D:/Coding/DeepLearning/Datasets",
                                                    train=False, download=True, transform=transform)
            test_loader = data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
            # classes = ("plane","car","bird","cat","deer","dog","frog","horse","ship","truck")
        case "tiny-imagenet-200":
            print("Using dataset tiny-imagenet-200")
            transform = trans.Compose([trans.ToTensor(), trans.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))])
            train_set = torchvision.datasets.ImageFolder(root="D:/Coding/DeepLearning/Datasets/tiny-imagenet-200/train", transform=transform)
            test_set = torchvision.datasets.ImageFolder(root="D:/Coding/DeepLearning/Datasets/tiny-imagenet-200/val", transform=transform)
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
        case _:
            raise Exception("Unrecognized dataset")
    
    
    # Instantiate model
    model = resnet.Net(CLASSES_NUM)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use CUDA if available
    model.to(device)
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

    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.5)


    # Train model and record training process
    print("----------------Training begins----------------")
    batch_group = 0
    
    train_batch_groups = []
    train_losses = []
    train_accuracies = []
    
    valid_batch_groups = []
    valid_losses = []
    valid_accuracies = []

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}:")
        total = 0
        correct = 0
        running_loss = 0.0
        
        for i, (items, labels) in enumerate(train_loader, 0):
            optimizer.zero_grad()
            
            outputs = model(items)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() # Accumulate loss
            _, indexes = torch.max(outputs.data, dim=1) # Manually retrive predicted class index from outputs
            correct += (indexes == labels).sum().item() # Count the number of correct predictions
            total += labels.size(0)
            
            if i % BATCH_GROUP_NUM == (BATCH_GROUP_NUM - 1): # Record training progress in the past [BATCH_GROUP_NUM] batches
                batch_group += 1
                train_batch_groups.append(batch_group)
                print(f'[Epoch: {epoch + 1}, items trained: {i + 1}] ' +
                      f'average loss: {running_loss / BATCH_GROUP_NUM:.3f}, ' + 
                      f'accuracy: {correct / total * 100:.3f}%')
                train_losses.append(running_loss / BATCH_GROUP_NUM)
                train_accuracies.append(correct / total * 100)
                running_loss = 0.0
                total = 0
                correct = 0
        
        valid_loss, valid_accuracy = validate(model, test_loader, criterion, device) # Validate after every epoch
        valid_batch_groups.append(batch_group)
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_accuracy)

    print("----------------Training completed----------------")
    print("Saving model to disk...")


    # Save model to disk
    PATH = 'D:/Coding/DeepLearning/ModelSaves/' + MODEL_NAME + '-' + DATASET_NAME + '.pth'
    torch.save(model.state_dict(), PATH)


    # Load model from disk (just for demonstration)
    # print("Loading model from disk...")
    # model = resnet.Net(CLASSES_NUM)
    # model.load_state_dict(torch.load(PATH, weights_only=True))
    
    
    # Plot training and validating process
    fig = pt.figure()
    
    loss_graph = fig.add_subplot(121)
    loss_graph.plot(train_batch_groups, train_losses, label = "Training Loss")
    loss_graph.plot(valid_batch_groups, valid_losses, label = "Validation Loss")
    loss_graph.set_title("Loss")
    loss_graph.set_xlabel("Batch Group Num")
    loss_graph.legend()
    
    accuracy_graph = fig.add_subplot(122)
    accuracy_graph.plot(train_batch_groups, train_accuracies, label = "Training Accuracy")
    accuracy_graph.plot(valid_batch_groups, valid_accuracies, label = "Validation Accuracy")
    accuracy_graph.set_title("Accuracy")
    accuracy_graph.set_xlabel("Batch Group Num")
    accuracy_graph.legend()
    
    pt.show()