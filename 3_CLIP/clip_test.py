import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image1 = preprocess(Image.open("D:/Coding/DeepLearning/ExaiPreparations/3_CLIP/clip_test_1.jpg")).unsqueeze(0)
image2 = preprocess(Image.open("D:/Coding/DeepLearning/ExaiPreparations/3_CLIP/clip_test_2.jpg")).unsqueeze(0)
image = torch.cat((image1, image2), dim = 0).to(device)
text = clip.tokenize(["a tiger", "a lion", "a goldfish", "a tuna"]).to(device)
print(image.shape)
print(text.shape)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    print(image_features.shape)
    print(text_features.shape)
    
    logits_per_image, logits_per_text = model(image, text)
    print(logits_per_image.shape)
    print(logits_per_text.shape)
    
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:\n", probs)