import torch
from EngFra import *
from Transformer import *



MODEL_NAME = "Transformer"
DATASET_NAME = "EnFr"
ROOT_PATH = "/home/hqdeng7/syang/ExaiPreparations/" # Path of ExaiPreparations
MODEL_PATH = ROOT_PATH + "ModelSaves/"
OUTPUT_PATH = ROOT_PATH + "Outputs/"

# if torch.cuda.is_available(): # Use CUDA if available
#     DEVICE = torch.device("cuda:4")
#     print("Using device: " + torch.cuda.get_device_name(DEVICE))
# else:
#     DEVICE = torch.device("cpu")
#     print("Using device: CPU")
    
DEVICE = torch.device("cpu") # just use CPU for now
MAX_LEN = 20
BATCH_SIZE = 8



# Create dataloader
dataset, train_loader, valid_loader = get_dataloaders(MAX_LEN, BATCH_SIZE, 0.9)

# Create model
model = Transformer(
    input_vocab_size=dataset.input_lang.n_words,
    output_vocab_size=dataset.output_lang.n_words,
    embed_dim=512,
    ffn_num_hiddens=1024,
    num_heads=8,
    num_layers=8,
    dropout=0.3
)

# Load model from disk
print("Loading model from disk...")
path = MODEL_PATH + MODEL_NAME + '-' + DATASET_NAME + '.pth'
model.load_state_dict(torch.load(path, weights_only=True))

# Evaluate on training dataset (with teacher-forcing)
input, input_len, target_S, target_E = next(iter(train_loader))
output = model(input, target_S, input_len, MAX_LEN)
predicted = output.argmax(dim=-1) # Greedily sample from distribution: (b, n)

for i in range(BATCH_SIZE):
    print(f"\nSample {i}:")
    print("Input:", [dataset.decode(index.item(), En=True) for index in input[i]])
    print("Predicted:", [dataset.decode(index.item(), En=False) for index in predicted[i]])
    print("Actual:", [dataset.decode(index.item(), En=False) for index in target_E[i]])
