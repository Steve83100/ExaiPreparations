from TinyImageNet import get_data
import math
from matplotlib import pyplot as plt
import time
import torch
import torch.nn as nn
import clip



# ====================================================================

# Hyperparameters

ROOT_PATH = "~/syang/ExaiPreparations/" # Path of ExaiPreparations
MODEL_PATH = ROOT_PATH + "ModelSaves/"
OUTPUT_PATH = ROOT_PATH + "Outputs/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 12
EPOCHS = 3



# ====================================================================

# Load CLIP model

model, preprocess = clip.load("ViT-B/32", device=DEVICE)
model.to(DEVICE)
# "preprocess" is composed of [Resize(224), CenterCrop(224), to_rgb(), ToTensor(), Normalize(mean, std)]
# Therefore we don't need additional normalizing, nor ToTensor() operations
# But we can still apply other augmentation such as random flips



# ====================================================================

# Define training and testing functions

def train(loader, model, text_features, device, criterion, optimizer):
    """Train model on given dataloader for 1 epoch. Returns average loss and accuracy."""
    model.train()
    total_loss = 0
    total_correct = 0
    total = 0
    optimizer.zero_grad()
    
    for images, label_ids in loader:
        images = images.to(device)
        image_features = model.encode_image(images) # images already preprocessed during dataset creation
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = (100.0 * image_features @ text_features.T).softmax(dim=-1) # logits: (batch_size, num_classes)
        
        loss = criterion(logits, label_ids) # label_ids: (batch_size)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        _, preds = torch.max(logits, dim = 1) # Manually retrive predicted class index from outputs
        total_correct += (preds == label_ids).sum().item() # Count the number of correct predictions
        total += label_ids.size(0)
        
    accuracy = total_correct / total * 100
        
    return total_loss / len(loader), accuracy

@torch.no_grad()
def test(loader, model, text_features, device, criterion):
    """Test model on given dataloader. Returns average loss and accuracy."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total = 0
    
    for images, label_ids in loader:
        images = images.to(device)
        image_features = model.encode_image(images) # images already preprocessed during dataset creation
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = (100.0 * image_features @ text_features.T).softmax(dim=-1) # logits: (batch_size, num_classes)
        
        loss = criterion(logits, label_ids) # label_ids: (batch_size)
        total_loss += loss.item()
        
        _, preds = torch.max(logits, dim = 1) # Manually retrive predicted class index from outputs
        total_correct += (preds == label_ids).sum().item() # Count the number of correct predictions
        total += label_ids.size(0)
        
    accuracy = total_correct / total * 100
        
    return total_loss / len(loader), accuracy

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)



# ====================================================================

# Get dataset and dataloaders

dataset, train_loader, valid_loader, test_loader = get_data(BATCH_SIZE, preprocess, [0.8, 0.1, 0.1])



# ====================================================================

# For all class ids: map to natural language, create prompt, tokenize, calculate representations

class_codes = dataset.original_dataset.classes
class_names = ["a photo of a " + dataset.code_name_dict[class_code] for class_code in class_codes]
# print(class_names)
class_tokens = clip.tokenize(class_names)
class_tokens.to(DEVICE)
# print(class_tokens.shape)
with torch.no_grad():
    text_features = model.encode_text(class_tokens)
text_features = text_features / text_features.norm(dim=-1, keepdim=True)
# print(text_features.shape)



# ====================================================================

# 1. Zero-shot

# print("Testing zero-shot performance...")
# loss, accu = test(test_loader, model, text_features, DEVICE, nn.CrossEntropyLoss())
# print("Zero-shot accuracy:", accu)



# ====================================================================

# 2. Fine-tune image encoder's projection head

print("Preparing to fine-tune image encoder's projection head...")

# Freeze everything except image encoder's projection head
for param in model.parameters():
    param.requires_grad = False
model.visual.proj.requires_grad = True
model.logit_scale.requires_grad = True

# Check parameters to fine-tune
for name, param in model.named_parameters():
    if param.requires_grad:
        print("Trainable:", name)

# Create optimizer on trainable parameters
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4, # Bigger lr for projection head fine-tuning
    weight_decay=1e-4
)

# Fine-tune
print("Start fine-tuning projection head...")
epochs = []
train_losses = []
train_accuracies = []
valid_losses = []
valid_accuracies = []
criterion = nn.CrossEntropyLoss()
start_time = time.time()

for epoch in range(1, EPOCHS):
    train_loss, train_accu = train(train_loader, model, text_features, DEVICE, criterion, optimizer)
    valid_loss, valid_accu = test(valid_loader, model, text_features, DEVICE, criterion)
    
    time_spent = asMinutes(time.time() - start_time)
    print(f"Epoch[{epoch}/{EPOCHS}]: Time spent:{time_spent}."
          + f"\n\tTrain: loss {train_loss:.4f}, accuracy {train_accu:.4f}"
          + f"\n\tValid: loss {valid_loss:.4f}, accuracy {valid_accu:.4f}")
    
    epochs.append(epoch)
    train_losses.append(train_loss)
    train_accuracies.append(train_accu)
    valid_losses.append(valid_loss)
    valid_accuracies.append(valid_accu)
    
# Plot graph
print("Plotting loss and accuracy graph...")
fig = plt.figure()

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

# plt.savefig(OUTPUT_PATH + "CLIP-microImageNet50.png")
plt.show()



# ====================================================================

# 3. Fine-tune image encoder's last K layers

# K = 3
# print("Preparing to fine-tune image encoder's last K layers...")

# # Freeze everything except image encoder's last K layers
# for param in model.parameters():
#     param.requires_grad = False
# blocks = model.visual.transformer.resblocks
# for block in blocks[-K:]:
#     for param in block.parameters():
#         param.requires_grad = True
# model.visual.proj.requires_grad = True
# model.logit_scale.requires_grad = True
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print("Trainable:", name)

# Create optimizer on trainable parameters
# optimizer = torch.optim.AdamW(
#     filter(lambda p: p.requires_grad, model.parameters()),
#     lr=1e-5, # Smaller lr for last-K fine-tuning
#     weight_decay=1e-4
# )
