import json
import torch
from sentence_transformers import SentenceTransformer
from sklearn import random_projection


all_tasks = json.load(open('/home/cjz/SFTonGFM/reshape/task_text.json', "r"))
all_tasks_embedding = {}
bert_model = SentenceTransformer('all-MiniLM-L6-v2')
for t_type in all_tasks.keys():
    task_text = all_tasks[t_type]
    task_embedding = bert_model.encode(task_text)
    task_embedding = task_embedding.reshape(1, -1)
    transformer = random_projection.GaussianRandomProjection(n_components=128, random_state=42)
    task_embedding = transformer.fit_transform(task_embedding)
    task_embedding = torch.tensor(task_embedding).float()
    all_tasks_embedding[t_type] = task_embedding.tolist()
torch.save(all_tasks_embedding, '/home/cjz/SFTonGFM/reshape/task_embedding.pt')
print('done')
