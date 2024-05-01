import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from graphgpt.conversation import conv_templates, SeparatorStyle
from graphgpt.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
from graphgpt.model import *
from graphgpt.model.graph_layers import MPNN
from graphgpt.model.utils import KeywordsStoppingCriteria
from torch_geometric.data import Data
import json
import copy
import numpy as np
import faiss
from torch.nn.parameter import Parameter
import datetime

import os
import requests
from PIL import Image
from io import BytesIO

from tqdm import tqdm
import json
import os.path as osp

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
wiki_categories = [
    'distributed computing architecture', 
    'operating systems', 
    'databases', 
    'computer security', 
    'internet protocols', 
    'programming language topics', 
    'computational linguistics', 
    'computer architecture', 
    'computer file systems', 
    'web technology',
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


def KNN_cos(train_set, test_set, n_neighbours):
    index = faiss.IndexFlatIP(train_set.shape[1])
    index.add(train_set)
    D, I = index.search(test_set, n_neighbours)
    return D, I

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
    
    args.use_rag = False
    if args.tuned_graph_tower == False:
        if args.gnn_type == 'gt':
            clip_graph, args_graph= load_model_pretrained(CLIP, '/home/cjz/GraphGPT/clip_gt_arxiv')
            graph_tower = graph_transformer(args_graph)
            graph_tower = transfer_param_tograph(clip_graph, graph_tower)
        elif args.gnn_type == 'mpnn':
            graph_tower = MPNN(in_channels = 128, hidden_channels = 256, out_channels = 128, dropout = 0.1, num_layers = 2, if_param = False)
    else:
        clip_graph, args_graph= load_model_pretrained(CLIP, '/home/cjz/GraphGPT/clip_gt_arxiv')
        graph_tower = graph_transformer(args_graph)
        graph_tower_dict = torch.load(os.path.join(args.graph_tower_path, 'tuned_graph_tower.bin'))
        new_keys = [k.replace('model.graph_tower.', '') for k in graph_tower_dict.keys()]
        modified_dict = dict(zip(new_keys, graph_tower_dict.values()))
        graph_tower.load_state_dict(modified_dict)
    train_file = load_prompting_file('/home/cjz/GraphGPT/reshape/train_items.json')
    if os.path.exists(os.path.join(args.embedding_file, 'train_embedding_array.npy')) and args.use_rag == True:
        train_embedding_array = np.load(os.path.join(args.embedding_file, 'train_embedding_array.npy'), allow_pickle=True)
    else:
        train_embedding_array = np.array([])
    if(args.use_rag == True):
        train_embedding_list = []
        print(f'Total: {len(train_file)}')
        for idx, instruct_item in tqdm(enumerate(train_file)):
            # instruct_item = prompt_file[0]
            # if idx >= 3: 
            #     break
            graph_dict = load_graph(instruct_item, args.graph_data_path)
            graph_data = graph_dict['graph_data']
            # label = instruct_item['conversations'][1]['value']
            node_forward_out = graph_tower(graph_data)
            train_embedding_list.append(node_forward_out[0].detach())
        train_embedding_list = torch.stack(train_embedding_list)
        train_embedding_array = train_embedding_list.numpy()
        np.save(os.path.join(args.embedding_file, 'train_embedding_array.npy'), train_embedding_array)
        
    eval_model(args, prompt_file, args.start_id, args.end_id, graph_tower, train_embedding_array, train_file)
    # for idx in range(len(idx_list) - 1):
    #     start_idx = idx_list[idx]
    #     end_idx = idx_list[idx + 1]
        
    #     start_split = split_list[idx]
    #     end_split = split_list[idx + 1]
    #     eval_model(args, prompt_file[start_idx:end_idx], start_split, end_split)
    #     ans_handles.append(
    #         eval_model.remote(
    #             args, prompt_file[start_idx:end_idx], start_split, end_split
    #         )
    #     )

    # ans_jsons = []
    # for ans_handle in ans_handles:
    #     ans_jsons.extend(ray.get(ans_handle))

    # with open(args.output_res_path, "w") as ans_file:
    #     for line in ans_jsons:
    #         ans_file.write(json.dumps(line) + "\n")


# @ray.remote(num_gpus=1)
@torch.inference_mode()
def eval_model(args, prompt_file, start_idx, end_idx, graph_tower, train_embedding_array, train_file):
    # load prompting file
    # prompt_file = load_prompting_file(args.prompting_file)


    # Model
    disable_torch_init()
    # model_name = os.path.expanduser(args.model_name)
    print('start loading')
    # tokenizer = AutoTokenizer.from_pretrained("/home/cjz/checkpoints/few-shot-prompt")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print('finish loading')

    print('start loading')
    model = GraphLlamaForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16, use_cache=True, low_cpu_mem_usage=True).cuda()
    print('finish loading')

    use_graph_start_end = getattr(model.config, "use_graph_start_end", False)
    tokenizer.add_tokens([DEFAULT_GRAPH_PATCH_TOKEN], special_tokens=True)
    if use_graph_start_end:
        tokenizer.add_tokens([DEFAULT_G_START_TOKEN, DEFAULT_G_END_TOKEN], special_tokens=True)
    
    if args.combined_graph_prompt == True:
        prompt_dict = torch.load(os.path.join(args.tuned_graph_prompt, "combined_graph_prompt.bin")) 
        
        new_linear_dict = {
            'weight': prompt_dict['model.new_prompt_linear.weight'].half(),
            'bias': prompt_dict['model.new_prompt_linear.bias'].half()
        }
        new_prompt_linear = torch.nn.Linear(128, 128)
        new_prompt_linear.load_state_dict(new_linear_dict)
        new_prompt_linear = new_prompt_linear.half()
        model.get_model().new_prompt_linear = new_prompt_linear.cuda()
        
        frozen_linear_dict = {
            'weight': prompt_dict['model.frozen_prompt_linear.weight'].half(),
            'bias': prompt_dict['model.frozen_prompt_linear.bias'].half()
        }
        frozen_prompt_linear = torch.nn.Linear(128, 128)
        frozen_prompt_linear.load_state_dict(frozen_linear_dict)
        frozen_prompt_linear = frozen_prompt_linear.half()
        model.get_model().frozen_prompt_linear = frozen_prompt_linear.cuda()
        
        model.get_model().new_prompt_weight = Parameter(prompt_dict['model.new_prompt_weight'].cuda())
        model.get_model().new_prompt_weight.to(device='cuda', dtype=torch.float16)
        
        model.get_model().forzen_prompt_weight = Parameter(prompt_dict['model.frozen_prompt_weight'].cuda())
        model.get_model().forzen_prompt_weight.to(device='cuda', dtype=torch.float16)   
        
        model.get_model().alpha_linear = Parameter(prompt_dict['model.alpha_linear'])
        model.get_model().alpha_linear.to(device='cuda', dtype=torch.float16)
        model.get_model().alpha_weight = Parameter(prompt_dict['model.alpha_weight'])
        model.get_model().alpha_weight.to(device='cuda', dtype=torch.float16)  
        
    elif args.tuned_graph_prompt != None:
        prompt_dict = torch.load(os.path.join(args.tuned_graph_prompt, "tuned_graph_prompt.bin"))
        linear_dict = {
            'weight': prompt_dict['model.prompt_linear.weight'].half(),
            'bias': prompt_dict['model.prompt_linear.bias'].half()
        }
        prompt_linear = torch.nn.Linear(128, 128)
        prompt_linear.load_state_dict(linear_dict)
        model.get_model().prompt_weight = Parameter(prompt_dict['model.prompt_weight'].cuda())
        model.get_model().prompt_weight.to(device='cuda', dtype=torch.float16)
        prompt_linear =  prompt_linear.half()
        model.get_model().prompt_linear = prompt_linear.cuda()
        

    # print(model)
    # graph_tower = model.get_model().graph_tower
    
    # TODO: add graph tower
    # if graph_tower.device.type == 'meta':
    #     print('meta')
    
    model.get_model().graph_tower = graph_tower.cuda()
    # else:
    #     print('other')
    # print(next(graph_tower.parameters()).dtype)
    graph_tower.to(device='cuda', dtype=torch.float16)
    graph_config = graph_tower.config
    graph_config.graph_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_GRAPH_PATCH_TOKEN])[0]
    graph_config.use_graph_start_end = use_graph_start_end
    if use_graph_start_end:
        graph_config.graph_start_token, graph_config.graph_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_G_START_TOKEN, DEFAULT_G_END_TOKEN])
    # TODO: add graph token len

    res_data = []
    print(f'total: {len(prompt_file)}')
    correct = 0
    almost_correct = 0
    for idx, instruct_item in tqdm(enumerate(prompt_file)):
        # instruct_item = prompt_file[0]
        # if idx >= 3: 
        #     break
        graph_dict = load_graph(instruct_item, args.graph_data_path)
        graph_token_len = graph_dict['graph_token_len']
        graph_data = graph_dict['graph_data']

        qs = instruct_item["conversations"][0]["value"]
        # if use_graph_start_end:
        #     qs = qs + '\n' + DEFAULT_G_START_TOKEN + DEFAULT_GRAPH_PATCH_TOKEN * graph_token_len + DEFAULT_G_END_TOKEN
        # else:
        #     qs = qs + '\n' + DEFAULT_GRAPH_PATCH_TOKEN * graph_token_len

        replace_token = DEFAULT_GRAPH_PATCH_TOKEN * graph_token_len
        replace_token = DEFAULT_G_START_TOKEN + replace_token + DEFAULT_G_END_TOKEN
        qs = qs.replace(DEFAULT_GRAPH_TOKEN, replace_token)
        
        

        # if "v1" in args.model_name.lower():
        #     conv_mode = "graphchat_v1"
        # else: 
        #     raise ValueError('Don\'t support this model')
        conv_mode = "rag"

        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
        else:
            args.conv_mode = conv_mode

        conv = conv_templates[args.conv_mode].copy()
        graph_data_list = []
        rag_label = ''
        new_qs = ''
        if args.use_rag:
            graph_data.graph_node = graph_data.graph_node.half()
            node_forward_out = graph_tower(graph_data)
            target_embedding = node_forward_out[0].cpu().detach().numpy().astype('float32')
            target_embedding = target_embedding.reshape(1, 128)
            train_embedding = train_embedding_array.astype('float32')
            faiss.normalize_L2(target_embedding)
            faiss.normalize_L2(train_embedding)
            distance, index = KNN_cos(train_embedding, target_embedding, 5)
            new_qs = 'Here are some examples: \n'
            count = 1
            for id in index[0]:
                example_graph_dict = load_graph(train_file[id], args.graph_data_path)
                example_graph_token_len = example_graph_dict['graph_token_len']
                example_graph_data = example_graph_dict['graph_data']
                example_graph_data.graph_node = example_graph_data.graph_node.to(torch.float16)
                graph_data_list.append(example_graph_data.cuda())
                # node_feature = torch.tensor(train_embedding_array[id])
                # node_feature = model.get_model().graph_projector(node_feature)
                replace_token = DEFAULT_GRAPH_PATCH_TOKEN * example_graph_token_len
                replace_token = DEFAULT_G_START_TOKEN + replace_token + DEFAULT_G_END_TOKEN
                example = train_file[id]['conversations'][0]['value'].replace(DEFAULT_GRAPH_TOKEN, replace_token)
                new_qs = new_qs + f'Example {count}: \n' + example + '\nAnswer: ' + train_file[id]['conversations'][1]['value'] + '\n'
                count += 1
                rag_label = rag_label + train_file[id]['conversations'][1]['value'] + '; '
            new_qs = new_qs + 'Please note that the papers I have selected as examples are very similar to the one you need to categorize. You can use the classifications from my samples as a reference.' + 'The one you need to categorize: \n'
        new_qs = new_qs + qs
            
        conv.append_message(conv.roles[0], new_qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        inputs = tokenizer([prompt])

        

        input_ids = torch.as_tensor(inputs.input_ids).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        graph_data.graph_node = graph_data.graph_node.to(torch.float16)
        # graph_data.edge_index = graph_data.edge_index.to(torch.float16)
        graph_data_list.append(graph_data.cuda())
        
        try:
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    graph_data=graph_data.cuda(),
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=1024,
                    stopping_criteria=[stopping_criteria])
        except RuntimeError as e:
            if "probability tensor contains either `inf`, `nan` or element < 0" in str(e):
                print(f"Skipping data due to error: {e}")
                continue
            else:
                raise e
            
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        # print(outputs)
        common, res, label, almost_common = find_common_categories(outputs, instruct_item['conversations'][1]['value'], wiki_categories)
        
        if len(common) > 0:
            correct += 1
        elif len(almost_common) != 0:
            almost_correct += 1

        res_data.append({"id": instruct_item["id"], "node_idx": instruct_item["graph"]["node_idx"], "res": res, "outputs": outputs, 
                         'label': label, 'common': list(common), 'almost common': almost_common, 'rag label': rag_label}.copy())
        # with open(osp.join(args.output_res_path, 'arxiv_test_res_{}_{}_with_prompt.json'.format(start_idx, end_idx)), "w") as fout:
        #     json.dump(res_data, fout, indent=4)
    print('acc = ', correct/len(prompt_file))
    print('almost correct answer: ', almost_correct)
    lead_dict = {
        'acc': correct/len(prompt_file),
        'almost_correct': almost_correct,
    }
    res_data.insert(0, lead_dict)
    current_datetime = datetime.datetime.now()
    with open(osp.join(args.output_res_path, 'cora_tit_gen_{}_{}.json'.format(len(prompt_file), current_datetime)), "w") as fout:
        json.dump(res_data, fout, indent=4)
    return res_data
    # with open(args.output_res_path, "w") as fout:
    #     json.dump(res_data, fout, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    # parser.add_argument("--image-file", type=str, required=True)
    # parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--prompting_file", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--graph_data_path", type=str, default=None)

    parser.add_argument("--output_res_path", type=str, default=None)
    parser.add_argument("--num_gpus", type=int, default=4)

    parser.add_argument("--start_id", type=int, default=0)
    parser.add_argument("--end_id", type=int, default=20567)
    
    parser.add_argument('--tuned_graph_tower', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--graph_tower_path", type=str, default=None)
    parser.add_argument("--gnn_type", type=str, default="gt")
    parser.add_argument("--use_rag", type=bool, default=False)
    parser.add_argument("--embedding_file", type=str, default='/home/cjz/GraphGPT/reshape')
    parser.add_argument("--tuned_graph_prompt", type=str, default=None)
    parser.add_argument("--combined_graph_prompt", type=bool, default=True)

    args = parser.parse_args()

    # eval_model(args)

    # ray.init()
    run_eval(args, args.num_gpus)


# protobuf             4.22.3