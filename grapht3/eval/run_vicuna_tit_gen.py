import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from graphgpt.conversation import conv_templates, SeparatorStyle
from graphgpt.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
from graphgpt.model import *
from graphgpt.model.utils import KeywordsStoppingCriteria
from torch_geometric.data import Data
import json
import copy
import re
from collections import defaultdict
from nltk.translate.bleu_score import sentence_bleu

import os
import os.path as osp
import requests
from PIL import Image
from io import BytesIO

from tqdm import tqdm
import json

import ray

# os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

DEFAULT_GRAPH_TOKEN = "<graph>"
DEFAULT_GRAPH_PATCH_TOKEN = "<g_patch>"
DEFAULT_G_START_TOKEN = "<g_start>"
DEFAULT_G_END_TOKEN = "<g_end>"

categories = [
    "artificial intelligence, agents",
    "artificial intelligence, data mining",
    "artificial intelligence, expert systems",
    "artificial intelligence, games and search",
    "artificial intelligence, knowledge representation",
    "artificial intelligence, machine learning, case-based",
    "artificial intelligence, machine learning, genetic algorithms",
    "artificial intelligence, machine learning, neural networks",
    "artificial intelligence, machine learning, probabilistic methods",
    "artificial intelligence, machine learning, reinforcement learning",
    "artificial intelligence, machine learning, rule learning",
    "artificial intelligence, machine learning, theory",
    "artificial intelligence, nlp",
    "artificial intelligence, planning",
    "artificial intelligence, robotics",
    "artificial intelligence, speech",
    "artificial intelligence, theorem proving",
    "artificial intelligence, vision and pattern recognition",
    "data structures, algorithms and theory, computational complexity",
    "data structures, algorithms and theory, computational geometry",
    "data structures, algorithms and theory, formal languages",
    "data structures, algorithms and theory, hashing",
    "data structures, algorithms and theory, logic",
    "data structures, algorithms and theory, parallel",
    "data structures, algorithms and theory, quantum computing",
    "data structures, algorithms and theory, randomized",
    "data structures, algorithms and theory, sorting",
    "databases, concurrency",
    "databases, deductive",
    "databases, object oriented",
    "databases, performance",
    "databases, query evaluation",
    "databases, relational",
    "databases, temporal",
    "encryption and compression, compression",
    "encryption and compression, encryption",
    "encryption and compression, security",
    "hardware and architecture, distributed architectures",
    "hardware and architecture, high performance computing",
    "hardware and architecture, input output and storage",
    "hardware and architecture, logic design",
    "hardware and architecture, memory structures",
    "hardware and architecture, microprogramming",
    "hardware and architecture, vlsi",
    "human computer interaction, cooperative",
    "human computer interaction, graphics and virtual reality",
    "human computer interaction, interface design",
    "human computer interaction, multimedia",
    "human computer interaction, wearable computers",
    "information retrieval, digital library",
    "information retrieval, extraction",
    "information retrieval, filtering",
    "information retrieval, retrieval",
    "nan",
    "networking, internet",
    "networking, protocols",
    "networking, routing",
    "networking, wireless",
    "operating systems, distributed",
    "operating systems, fault tolerance",
    "operating systems, memory management",
    "operating systems, realtime",
    "programming, compiler design",
    "programming, debugging",
    "programming, functional",
    "programming, garbage collection",
    "programming, java",
    "programming, logic",
    "programming, object oriented",
    "programming, semantics",
    "programming, software development"
]

def find_common_categories(output, label, categories):
    # 转换为小写以进行不区分大小写的匹配
    output = output.lower()
    label = label.lower()

    # 找到每个字符串中的分类
    output_categories = [cat for cat in categories if cat in output]
    label_categories = [cat for cat in categories if cat in label]

    # 查找共同分类
    common_categories = set(output_categories).intersection(label_categories)
    
    almost_common = []
    # 分割字符串并去除最后一个词
    for output_category in output_categories:
        parts1 = output_category.rsplit(',', 1)[0]
        for label_category in label_categories:
            parts2 = label_category.rsplit(',', 1)[0]
            if parts1 == parts2:
                almost_common.append(parts1)

    return common_categories, output_categories, label_categories, almost_common

def load_graph(instruct_item, graph_data_path): 
    graph_data_all = torch.load(graph_data_path)
    graph_dict = instruct_item['graph']
    graph_edge_index = torch.Tensor(copy.deepcopy(graph_dict['edge_index'])).long()
    graph_node_list = copy.deepcopy(graph_dict['node_list'])
    target_node = copy.deepcopy(graph_dict['node_idx'])
    graph_type = copy.deepcopy(instruct_item['id']).split('_')[0]
    graph_node_rep = graph_data_all[graph_type].x[graph_node_list] ## 
    
    cur_token_len = len(graph_node_rep)   # FIXME: 14 is hardcoded patch size

    graph_ret = Data(graph_node = graph_node_rep, edge_index=graph_edge_index, target_node = torch.tensor([target_node]))

    return {
        'graph_data': graph_ret, 
        'graph_token_len': cur_token_len
    }


def load_prompting_file(file_path): 
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# def prepare_query(instruct_item): 


def run_eval(args, num_gpus):
    # split question file into num_gpus files
    prompt_file = load_prompting_file(args.prompting_file)
    prompt_file = prompt_file[args.start_id:args.end_id]
    train_file = load_prompting_file(args.train_file)
    # chunk_size = len(prompt_file) // num_gpus
    # ans_handles = []
    # split_list = list(range(args.start_id, args.end_id, chunk_size))
    # idx_list = list(range(0, len(prompt_file), chunk_size))
    # if len(split_list) == num_gpus: 
    #     split_list.append(args.end_id)
    #     idx_list.append(len(prompt_file))
    # elif len(split_list) == num_gpus + 1: 
    #     split_list[-1] = args.end_id
    #     idx_list[-1] = len(prompt_file)
    # else: 
    #     raise ValueError('error in the number of list')
    
    if osp.exists(args.output_res_path) is False: 
        os.mkdir(args.output_res_path)

    eval_model(args, prompt_file, train_file, args.start_id, args.end_id)

    # with open(args.output_res_path, "w") as ans_file:
    #     for line in ans_jsons:
    #         ans_file.write(json.dumps(line) + "\n")


# @ray.remote(num_gpus=1)
@torch.inference_mode()
def eval_model(args, prompt_file, train_file, start_id, end_id):
    # load prompting file
    # prompt_file = load_prompting_file(args.prompting_file)


    # Model

    disable_torch_init()
    # model_name = os.path.expanduser(args.model_name)
    print('start loading')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print('finish loading')

    print('start loading')
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16, use_cache=True, low_cpu_mem_usage=True).cuda()
    print('finish loading')

    
    print(f'total: {len(prompt_file)}')
    res_file = osp.join(args.output_res_path, f'cora_vicuna_test_res_{start_id}_{end_id}_5shots.json')
    res_data = []
    print('*'*10, 'create res file', '*'*10)
    with open(res_file, 'w') as f:
        json.dump(res_data, f)
    examples = []
    for idx, instruct_item in enumerate(train_file):
        qs = instruct_item['conversations'][0]['value']
        catgory = instruct_item['conversations'][1]['value']
        example = qs + '\n' + catgory
        examples.append(example)
            
    all_score_bleu_1, all_score_bleu_2, all_score_bleu_3, all_score_bleu_4 = 0, 0, 0, 0

    for idx, instruct_item in tqdm(enumerate(prompt_file)):
        # instruct_item = prompt_file[0]
        # if idx >= 3: 
        #     break

        qs = instruct_item["conversations"][0]["value"]
        ans = instruct_item["conversations"][1]["value"]
        
        pattern = r'<graph>'

        qs = re.sub(pattern, '', qs)

        conv_mode = "vicuna_v1_1"

        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
        else:
            args.conv_mode = conv_mode

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        if args.zero_shot == 'false':
            for example in examples[:args.shot_num]:
                prompt += '\n' + example
        input_ids = tokenizer([prompt]).input_ids

        output_ids = model.generate(
            torch.as_tensor(input_ids).cuda(),
            do_sample=True,
            temperature=0.7,
            max_new_tokens=100,
        )
        output_ids = output_ids[0][len(input_ids[0]) :]
        outputs = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        outputs = outputs.split()
        label = instruct_item["conversations"][1]["value"].split()
        bleu_1 = sentence_bleu([outputs], label, weights=(1, 0, 0, 0))
        bleu_2 = sentence_bleu([outputs], label, weights=(0.5, 0.5, 0, 0))
        bleu_3 = sentence_bleu([outputs], label, weights=(0.33, 0.33, 0.33, 0))
        bleu_4 = sentence_bleu([outputs], label, weights=(0.25, 0.25, 0.25, 0.25))
        # score = bleu_1 + bleu_2 + bleu_3 + bleu_4

        res_data.append({"id": instruct_item["id"], "node_idx": instruct_item["graph"]["node_idx"], "res": outputs}.copy())
        all_score_bleu_1 += bleu_1
        all_score_bleu_2 += bleu_2
        all_score_bleu_3 += bleu_3
        all_score_bleu_4 += bleu_4
    print('bleu_1 = ', all_score_bleu_1/len(prompt_file))
    print('bleu_2 = ', all_score_bleu_2/len(prompt_file))
    print('bleu_3 = ', all_score_bleu_3/len(prompt_file))
    print('bleu_4 = ', all_score_bleu_4/len(prompt_file))
    lead_dict = {
        'bleu_1': all_score_bleu_1/len(prompt_file),
        'bleu_2': all_score_bleu_2/len(prompt_file),
        'bleu_3': all_score_bleu_3/len(prompt_file),
        'bleu_4': all_score_bleu_4/len(prompt_file),
    }
    res_data.insert(0, lead_dict)
    with open(res_file, 'w') as f:
        json.dump(res_data, f)
    return res_data
    # with open(args.output_res_path, "w") as fout:
    #     json.dump(res_data, fout, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    # parser.add_argument("--image-file", type=str, required=True)
    # parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--prompting_file", type=str, default=None)
    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--graph_data_path", type=str, default=None)

    parser.add_argument("--output_res_path", type=str, default=None)
    parser.add_argument("--num_gpus", type=int, default=2)
    parser.add_argument("--zero_shot", action="store_true")

    parser.add_argument("--start_id", type=int, default=0)
    parser.add_argument("--end_id", type=int, default=20567)
    parser.add_argument("--shot_num", type=int, default=5)

    args = parser.parse_args()

    # eval_model(args)

    ray.init()
    run_eval(args, args.num_gpus)


# protobuf             4.22.3