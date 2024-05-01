import pandas as pd
import torch
from torch_geometric.utils import to_undirected
import json
from gensim.models import Word2Vec
import numpy as np

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

def get_data():
    products_text = pd.read_csv('/home/cjz/SFTonGFM/reshape_products/ogbn-products_subset.csv')
    products_graph = torch.load('/home/cjz/SFTonGFM/reshape_products/ogbn-products_subset.pt')
    adj_t = products_graph.adj_t
    adj_t_coo = adj_t.coo()
    edge_index = torch.stack((adj_t_coo[0], adj_t_coo[1]), dim=0)
    edge_index = to_undirected(edge_index)
    products_graph.edge_index = edge_index
    text_list = []
    for i in range(len(products_text)):
        text = 'Products title: ' + products_text.loc[0]['title'] + '\nProducts description: ' + products_text.loc[0]['content']
        text_list.append(text)
    node_text_feature_list = []
    model = Word2Vec(text_list, vector_size=128, window=5, min_count=1, workers=4)
    for i in range(len(products_text)):
        node_text_feature = document_embedding(model, text_list[i])
        node_text_feature_list.append(node_text_feature)
    products_graph.x = torch.tensor(node_text_feature_list, dtype=torch.float)
    print(products_graph.x.shape)
    
    all_conversations = []
    for i in range(len(products_text)):
        human_value = (f'Given a production purchase graph: \n<graph>\nwhere the 0th node is the target product,'
                       f'with the following information of target product: \n {text_list[i]} \n'
                       f'Question: Which of the following categories of cumputer science does this paper belong to: {label_text} ?'
                       'Directly give the full name of the most likely category of this paper.')
    
get_data()