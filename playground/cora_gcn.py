from graphgpt.model import *
import os
import json
import time
import torch
import argparse
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.logging import init_wandb, log


selected_categories = [
    "operating systems, memory management",
    "artificial intelligence, planning",
    "artificial intelligence, vision and pattern recognition",
    "artificial intelligence, machine learning, case-based",
    "artificial intelligence, agents",
    "artificial intelligence, machine learning, probabilistic methods",
    "operating systems, distributed",
    "artificial intelligence, machine learning, genetic algorithms",
    "human computer interaction, graphics and virtual reality",
    "programming, object oriented",
    "encryption and compression, encryption",
    "networking, protocols",
    "programming, software development",
    "programming, compiler design",
    "artificial intelligence, machine learning, theory",
    "artificial intelligence, machine learning, neural networks",
    "programming, logic",
    "operating systems, realtime",
    "artificial intelligence, speech",
    "artificial intelligence, robotics",
    "artificial intelligence, games and search"
]

categories = {}
for i in range(len(selected_categories)):
    categories[selected_categories[i]] = i

def load_prompting_file(file_path): 
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

graph_data_all = torch.load('/home/yjy/GraphGPT/graph_data/graph_data_all.pt')
graph = graph_data_all['cora']

train_file = load_prompting_file('/home/yjy/GraphGPT/reshape/train_items.json')
test_file = load_prompting_file('/home/yjy/GraphGPT/reshape/test_items.json')

# modify label
sample_id = []
sample_train_id = []
sample_test_id = []
for idx, instruct_item in enumerate(train_file):
    label = categories[instruct_item['conversations'][1]['value']]
    id = instruct_item['graph']['node_idx']
    graph.y[id] = label
    sample_id.append(id)
    sample_train_id.append(id)

for idx, instruct_item in enumerate(test_file):
    label = categories[instruct_item['conversations'][1]['value']]
    id = instruct_item['graph']['node_idx']
    graph.y[id] = label
    sample_id.append(id)
    sample_test_id.append(id)

node_id = set([i for i in range(graph.num_nodes)])
sample_id = set(sample_id)
no_sample_id = list(node_id - sample_id)
for i in no_sample_id:
    graph.y[i] = 0

# modify mask
length = graph.num_nodes
graph.train_mask = torch.tensor([False for i in range(length)])
graph.test_mask = torch.tensor([False for i in range(length)])
for id in sample_train_id:
    graph.train_mask[id] = True
for id in sample_test_id:
    graph.test_mask[id] = True

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--hidden_channels', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=200)
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x

data = graph.to(device)
model = GCN(
    in_channels=data.num_features,
    hidden_channels=args.hidden_channels,
    out_channels=max(data.y.tolist())+1,
).to(device)

optimizer = torch.optim.Adam([
    dict(params=model.conv1.parameters(), weight_decay=5e-4),
    dict(params=model.conv2.parameters(), weight_decay=0)
], lr=args.lr)  # Only perform weight-decay on first convolution.

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_attr)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test():
    model.eval()
    pred = model(data.x, data.edge_index, data.edge_attr).argmax(dim=-1)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs

best_val_acc = test_acc = 0
times = []
for epoch in range(1, args.epochs + 1):
    start = time.time()
    loss = train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log(Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc, Test=test_acc)
    times.append(time.time() - start)
print(f'Median time per epoch: {torch.tensor(times).median():.4f}s')