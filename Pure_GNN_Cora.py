from graphgpt.model import *
import json
import torch
import os
import tqdm

selected_categories = {
    "operating systems, memory management": 0,
    "artificial intelligence, planning": 1,
    "artificial intelligence, vision and pattern recognition": 2,
    "artificial intelligence, machine learning, case-based": 3,
    "artificial intelligence, agents": 4,
    "artificial intelligence, machine learning, probabilistic methods": 5,
    "operating systems, distributed": 6,
    "artificial intelligence, machine learning, genetic algorithms": 7,
    "human computer interaction, graphics and virtual reality": 8,
    "programming, object oriented": 9,
    "encryption and compression, encryption": 10,
    "networking, protocols": 11,
    "programming, software development": 12,
    "programming, compiler design": 13,
    "artificial intelligence, machine learning, theory": 14,
    "artificial intelligence, machine learning, neural networks": 15,
    "programming, logic": 16,
    "operating systems, realtime": 17,
    "artificial intelligence, speech": 18,
    "artificial intelligence, robotics": 19,
    "artificial intelligence, games and search": 20,
}

def load_prompting_file(file_path): 
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

clip_graph, args_graph= load_model_pretrained(CLIP, '/home/cjz/GraphGPT/clip_gt_arxiv')
graph_tower = graph_transformer(args_graph)
graph_tower = transfer_param_tograph(clip_graph, graph_tower)

graph_data_all = torch.load('/home/cjz/GraphGPT/graph_data/graph_data_all.pt')
graph = graph_data_all['cora']

train_file = load_prompting_file('/home/cjz/GraphGPT/reshape/train_items.json')
test_file = load_prompting_file('/home/cjz/GraphGPT/reshape/test_items.json')

for idx, instruct_item in enumerate(train_file):
    label = instruct_item['conversations'][1]['value']
    id = instruct_item['graph']['node_idx']
    