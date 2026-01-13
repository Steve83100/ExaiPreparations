from TinyImageNet import get_data

import torch
import torch.nn as nn
import clip



# ====================================================================

# Hyperparameters

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 12
LR = 5e-5
EPOCHS = 50



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
    model.train()
    total_loss = 0
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
        
    return total_loss / len(loader)

@torch.no_grad()
def test(loader, model, text_features, device, criterion):
    model.eval()
    total_loss = 0
    
    for images, label_ids in loader:
        images = images.to(device)
        image_features = model.encode_image(images) # images already preprocessed during dataset creation
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = (100.0 * image_features @ text_features.T).softmax(dim=-1) # logits: (batch_size, num_classes)
        
        loss = criterion(logits, label_ids) # label_ids: (batch_size)
        total_loss += loss.item()
        
    return total_loss / len(loader)



# ====================================================================

# Get dataset and dataloaders

dataset, train_loader, valid_loader, test_loader = get_data(BATCH_SIZE, [0.8, 0.1, 0.1], preprocess)



# ====================================================================

# Prompt, tokenize and calculate representations for all classes

class_codes = dataset.dataset.classes
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

print(test(test_loader, model, text_features, DEVICE, nn.CrossEntropyLoss()))



# ====================================================================

# 2. Fine-tune projection head