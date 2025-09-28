import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torchvision
import torchvision.transforms as trans
import matplotlib.pyplot as pt
import os


# Create model structure
def vggBlock(num_convs, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.LazyConv2d(out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            vggBlock(1, 64),
            vggBlock(1, 128),
            vggBlock(2, 256),
            vggBlock(2, 512),
            nn.Flatten(),
            nn.LazyLinear(4096), nn.ReLU(),
            nn.LazyLinear(4096), nn.ReLU(),
            nn.LazyLinear(200) # No need to add final softmax. CrossEntropyLoss does it for us.
        )
        
    def forward(self, x):
        return self.net(x)



if __name__ == '__main__':
    batch_size = 10
    batch_group_num = 500 # Number of batches in a group. Will do some statistics after a group is trained
    
    # Retrive and transform data
    print("Fetching data...")
    transform = trans.Compose([trans.ToTensor(), trans.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))])
    trainSet = torchvision.datasets.ImageFolder(root="D:/Coding/DeepLearning/Datasets/tiny-imagenet-200/train", transform=transform)
    testSet = torchvision.datasets.ImageFolder(root="D:/Coding/DeepLearning/Datasets/tiny-imagenet-200/val", transform=transform)
    trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=batch_size, shuffle=True, pin_memory=True)
    testLoader = torch.utils.data.DataLoader(testSet, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    # Instantiate model
    print("Creating model...")
    model = Net()
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

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


    # Train model and record training process
    print("Training begins")
    batchGroup = 0
    total = 0
    correct = 0
    running_loss = 0.0
    batchGroupNum=[]
    trainLosses=[]
    # validLosses=[]
    trainAccuracies=[]
    # validAccuracies=[]

    for epoch in range(4):
        print(f"Starting epoch {epoch + 1}...")
        for i, (items, labels) in enumerate(trainLoader, 0):
            optimizer.zero_grad()
            
            outputs = model(items)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() # Accumulate loss
            _, indexes = torch.max(outputs.data, dim=1) # Manually retrive predicted class index from outputs
            correct += (indexes == labels).sum().item() # Count the number of correct predictions
            total += labels.size(0)
            
            if i % batch_group_num == (batch_group_num - 1): # print loss and accuracy every 2000 minibatches
                batchGroup += 1
                batchGroupNum.append(batchGroup)
                print(f'[epoch: {epoch + 1}, item:{i + 1:5d}] ' +
                      f'average loss: {running_loss / batch_group_num:.3f}, ' + 
                      f'accuracy: {correct / total * 100:.3f}')
                trainLosses.append(running_loss / batch_group_num)
                trainAccuracies.append(correct / total * 100)
                running_loss = 0.0
                total = 0
                correct = 0

    print('Training completed. Saving model to disk...')


    # Save model to disk
    PATH = 'D:/Coding/DeepLearning/ModelSaves/vgg_tinyImageNet.pth'
    torch.save(model.state_dict(), PATH)


    # Load model from disk (just for demonstration)
    print("Loading model from disk...")
    model = Net()
    model.load_state_dict(torch.load(PATH, weights_only=True))


    # Test model
    print("Testing...")
    total = 0
    correct = 0
    with torch.no_grad():
        for (items, labels) in testLoader:
            items, labels = items.to(device), labels.to(device)
            outputs = model(items)
            _, indexes = torch.max(outputs.data, dim=1)
            correct += (indexes == labels).sum().item()
            total += labels.size(0)
    print(f'Accuracy of the network on the 10000 test images: {correct / total * 100:.3f} %')