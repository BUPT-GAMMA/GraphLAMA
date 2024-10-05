import torch

from transformers import AutoConfig, StoppingCriteria


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = [tokenizer(keyword).input_ids for keyword in keywords]
        self.keyword_ids = [keyword_id[0] for keyword_id in self.keyword_ids if type(keyword_id) is list and len(keyword_id) == 1]
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
        else:
            for keyword_id in self.keyword_ids:
                if output_ids[0, -1] == keyword_id:
                    return True
            outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)[0]
            for keyword in self.keywords:
                if keyword in outputs:
                    return True
        return False

def concat_position_encoding(graph, node_features):
    num_nodes = node_features.size(0)
    position_encodings = torch.zeros((num_nodes, 2))  # 初始化位置编码
    
    # 目标节点位置编码
    position_encodings[0] = torch.tensor([0, 0])
    
    # 获取目标节点的所有一阶邻居
    target_node = 0
    neighbors_first_order = set(graph.edge_index[0, graph.edge_index[1] == target_node].tolist()) | set(graph.edge_index[1, graph.edge_index[0] == target_node].tolist())
    neighbors_first_order.discard(target_node)

    # 为一阶邻居设置位置编码
    for neighbor in neighbors_first_order:
        position_encodings[neighbor] = torch.tensor([0, 1])
    
    neighbors_second_order = set()
    for neighbor in neighbors_first_order:
        second_order = set(graph.edge_index[1, graph.edge_index[0] == neighbor].tolist()) | set(graph.edge_index[0, graph.edge_index[1] == neighbor].tolist())
        neighbors_second_order.update(second_order)
    
    # 排除目标节点和一阶邻居
    neighbors_second_order.difference_update(neighbors_first_order)
    neighbors_second_order.discard(target_node)
    
    # 为二阶邻居设置位置编码
    for neighbor in neighbors_second_order:
        position_encodings[neighbor] = torch.tensor([1, 0])
    
    position_encodings = position_encodings.to(device=node_features.device, dtype=node_features.dtype)
    # 将位置编码和原始节点特征进行拼接
    combined_features = torch.cat([node_features, position_encodings], dim=1)
    return combined_features
