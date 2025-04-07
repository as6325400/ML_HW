import torch
from models.model import ExampleCNN
from datasets.dataloader import make_test_dataloader
import argparse
import json
import os
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--weight', type=str, required=True, help='Path to the weight file')
parser.add_argument('--loss', type=str, default='ce', choices=['ce', 'bce', 'focal'], help='Loss function: ce / bce / focal')
parser.add_argument('--log', type=str, default='test_log.jsonl', help='Path to JSONL log file')
args = parser.parse_args()

base_path = os.path.dirname(os.path.abspath(__file__))
test_data_path = os.path.join(base_path, "data", "test")
weight_path = os.path.join(base_path, "weights", args.weight)

# load model and use weights we saved before
num_classes = 2 if args.loss == 'ce' else 1
model = ExampleCNN(num_classes)
model.load_state_dict(torch.load(weight_path, weights_only=True))
model = model.to(device)

# make dataloader for test data
test_loader = make_test_dataloader(test_data_path)

predict_correct = 0
model.eval()
with torch.no_grad():
    for data, target in tqdm(test_loader, desc="Testing"):
        data, target = data.to(device), target.to(device)

        output = model(data)

        if args.loss == 'ce':
            preds = output.data.max(1)[1]
        else:
            output = output.squeeze(1)
            preds = (torch.sigmoid(output) > 0.5).long()

        predict_correct += (preds == target).sum()
        
    accuracy = 100. * predict_correct / len(test_loader.dataset)
print(f'Test accuracy: {accuracy:.4f}%')

log_data = {
    "weight_path": weight_path,
    "accuracy": float(f"{accuracy:.4f}")
}
with open(args.log, "a") as f:
    f.write(json.dumps(log_data) + "\n")