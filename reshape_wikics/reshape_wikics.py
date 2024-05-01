import json
import os
import functools
import torch
from collections import defaultdict
from torch_geometric.datasets import WikiCS

from gensim.models import Word2Vec
import numpy as np

label_text = '1.Computational linguistics 2.Databases 3.Operating systems 3.Computer architecture 4.Computer security 5.Internet protocols 6.Computer file systems 7.Distributed computing architecture 8.Web technology 9.Programming language topics'
categories = {
    'Distributed computing architecture': 865, 
    'Operating systems': 2153, 
    'Databases': 667, 
    'Computer security': 2679, 
    'Internet protocols': 780, 
    'Programming language topics': 1424, 
    'Computational linguistics': 295, 
    'Computer architecture': 1933, 
    'Computer file systems': 413, 
    'Web technology': 492
}
# 训练Word2Vec模型
# model = Word2Vec(documents, vector_size=128, window=5, min_count=1, workers=4)

def load_file(file_path): 
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# 计算每个文档的平均词嵌入
def document_embedding(model, doc):
    # 过滤掉模型词汇表中不存在的词
    embeddings = [model.wv[word] for word in doc if word in model.wv]
    # 计算平均词嵌入
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(128)


def get_edges_of_node(pyg_data, node_id):
    edge_index = pyg_data.edge_index
    # Find the indices where source node ID or target node ID is 0
    indices = ((edge_index[0] == node_id) | (edge_index[1] == node_id)).nonzero(as_tuple=True)[0]
    # Get the edges
    edges = edge_index[:, indices]
    return edges

def convert_to_relative_ids(edge_index):
    # 从边的信息中提取出所有的节点
    node_list = torch.unique(edge_index)
    
    # 创建一个映射，将节点的绝对ID映射到其在node_list中的索引（即相对ID）
    id_mapping = {node_id.item(): index for index, node_id in enumerate(node_list)}
    
    # 使用映射将edge_index中的节点ID转换为相对节点ID
    relative_edge_index = edge_index.clone().detach()
    for edge in relative_edge_index.transpose(0, 1):
        edge[0] = id_mapping[edge[0].item()]
        edge[1] = id_mapping[edge[1].item()]
    
    return relative_edge_index, node_list

def get_data_orig(path):
    pyg_data = WikiCS(root=path)
    pyg_data = pyg_data[0]
    with open(os.path.join(path, "metadata.json")) as json_file:
        raw_data = json.load(json_file)
    node_info = raw_data["nodes"]
    label_info = raw_data["labels"]
    node_text_lst = []
    for node in node_info:
        node_text_lst.append(node['tokens'])
    node_text_feature_lst = []
    model = Word2Vec(node_text_lst, vector_size=128, window=5, min_count=1, workers=4)
    for node in node_info:
        node_text_feature = document_embedding(model, node['tokens'])
        node_text_feature_lst.append(node_text_feature)
    pyg_data.x = torch.tensor(node_text_feature_lst, dtype=torch.float)
    print(pyg_data.x.shape)
    
    # torch.save(pyg_data, os.path.join(path, "word2vec_data.pt"))
    all_conversations = []
    for id, node in enumerate(node_info):
        text = ((
                "wikipedia entry name: " + node["title"] + ". entry content: " + functools.reduce(
            lambda x, y: x + " " + y, node["tokens"])).lower().strip())
        human_value = (f'Given a wikipedia citation graph: \n<graph>\nwhere the 0th node is the target paper,'
                       f'with the following information: \n {text} \n'
                       f'Question: Which of the following categories of cumputer science does this paper belong to: {label_text} ?'
                       'Directly give the full name of the most likely category of this paper.')
        gpt_value = node['label']
        problem = [
            {'from': 'human', 'value': human_value},
            {'from': 'gpt', 'value': gpt_value}
        ]
        
        edge_index = get_edges_of_node(pyg_data, id)
        relative_edge_index, node_list = convert_to_relative_ids(edge_index)
        graph_dict = {
            "edge_index": relative_edge_index.tolist(),
            "node_list": node_list.tolist(),
            "node_idx": id
        }
        
        conversation_dict = {
            "id": f'Wiki_cs_{id}',
            "conversations": problem,
            "graph": graph_dict
        }
        all_conversations.append(conversation_dict)
    with open(os.path.join(path, "conversations.json"), 'w') as f:
        json.dump(all_conversations, f, ensure_ascii=False, indent=4)
    
def split(path):
    all_items = load_file(path)
    category_counts = defaultdict(int)
    test_set, train_5shots, train_20shots, train_50shots = [], [], [], []
    first_five_keys = list(categories.keys())[:5]
    for item in all_items:
        cat = item['conversations'][1]['value']
        if len(item['conversations'][0]['value']) > 4096:
            continue
        if cat in first_five_keys:
            qs = item['conversations'][0]['value']
            item['conversations'][0]['value'] = qs.replace(label_text, '1.Distributed computing architecture 2.Operating systems 3.Databases 4.Computer security 5.Internet protocols')
            category_counts[cat] += 1
            if category_counts[cat] <= 50:
                test_set.append(item)
            elif category_counts[cat] <= 55:
                train_5shots.append(item)
            elif category_counts[cat] <= 75:
                train_20shots.append(item)
            elif category_counts[cat] <= 125: 
                train_50shots.append(item)
        # category_counts[cat] += 1
        # if category_counts[cat] <= 50:
        #     test_set.append(item)
        # elif category_counts[cat] <= 55:
        #     train_5shots.append(item)
        # elif category_counts[cat] <= 75:
        #     train_20shots.append(item)
        # elif category_counts[cat] <= 125: 
        #     train_50shots.append(item)
    with open(path.replace('conversations', 'test_item_5ways'), 'w') as f:
        json.dump(test_set, f, ensure_ascii=False, indent=4)
    for i in [5, 20, 50]:
        with open(path.replace('conversations', f'train_{i}shots_5ways'), 'w') as f:
            json.dump(eval(f'train_{i}shots'), f, ensure_ascii=False, indent=4)
    print(len(test_set), len(train_5shots), len(train_20shots), len(train_50shots))
    
get_data_orig('/home/cjz/OneForAll/data/single_graph/wikics')
split('/home/cjz/OneForAll/data/single_graph/wikics/conversations.json')