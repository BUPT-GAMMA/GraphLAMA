import argparse
from tqdm import tqdm
import json
import os.path as osp
from collections import defaultdict
from grapht3.model import *
import re
import torch
import os
import random

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

selected_categories = {
    "operating systems, memory management": 112,
    "artificial intelligence, planning": 96,
    "artificial intelligence, vision and pattern recognition": 151,
    "artificial intelligence, machine learning, case-based": 65,
    "artificial intelligence, agents": 104,
    "artificial intelligence, machine learning, probabilistic methods": 88,
    "operating systems, distributed": 104,
    "artificial intelligence, machine learning, genetic algorithms": 88,
    "human computer interaction, graphics and virtual reality": 93,
    "programming, object oriented": 61,
    "encryption and compression, encryption": 82,
    "networking, protocols": 87,
    "programming, software development": 96,
    "programming, compiler design": 100,
    "artificial intelligence, machine learning, theory": 75,
    "artificial intelligence, machine learning, neural networks": 170,
    "programming, logic": 66,
    "operating systems, realtime": 72,
    "artificial intelligence, speech": 59,
    "artificial intelligence, robotics": 100,
    "artificial intelligence, games and search": 69,
}

five_ways_categories = {
    "operating systems, memory management": 112,
    "artificial intelligence, planning": 96,
    "artificial intelligence, vision and pattern recognition": 151,
    "artificial intelligence, machine learning, case-based": 65,
    "artificial intelligence, agents": 104,
    "artificial intelligence, machine learning, probabilistic methods": 88,
    "operating systems, distributed": 104,
    "artificial intelligence, machine learning, genetic algorithms": 88,
    "human computer interaction, graphics and virtual reality": 93,
    "programming, object oriented": 61,
    "encryption and compression, encryption": 82,
    "networking, protocols": 87,
    "programming, software development": 96,
    "programming, compiler design": 100,
    "artificial intelligence, machine learning, theory": 75,
    "artificial intelligence, machine learning, neural networks": 170,
    "programming, logic": 66,
    "operating systems, realtime": 72,
    "artificial intelligence, robotics": 100,
    "artificial intelligence, games and search": 69,
}

def load_file(file_path): 
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def reshape(args):
    selected_indices = []
    none_selected_indices = []
    old_string = '1. artificial intelligence, agents 2. artificial intelligence, data mining 3. artificial intelligence, expert systems 4. artificial intelligence, games and search 5. artificial intelligence, knowledge representation 6. artificial intelligence, machine learning, case-based 7. artificial intelligence, machine learning, genetic algorithms 8. artificial intelligence, machine learning, neural networks 9. artificial intelligence, machine learning, probabilistic methods 10. artificial intelligence, machine learning, reinforcement learning 11. artificial intelligence, machine learning, rule learning 12. artificial intelligence, machine learning, theory 13. artificial intelligence, nlp 14. artificial intelligence, planning 15. artificial intelligence, robotics 16. artificial intelligence, speech 17. artificial intelligence, theorem proving 18. artificial intelligence, vision and pattern recognition 19. data structures  algorithms and theory, computational complexity 20. data structures  algorithms and theory, computational geometry 21. data structures  algorithms and theory, formal languages 22. data structures  algorithms and theory, hashing 23. data structures  algorithms and theory, logic 24. data structures  algorithms and theory, parallel 25. data structures  algorithms and theory, quantum computing 26. data structures  algorithms and theory, randomized 27. data structures  algorithms and theory, sorting 28. databases, concurrency 29. databases, deductive 30. databases, object oriented 31. databases, performance 32. databases, query evaluation 33. databases, relational 34. databases, temporal 35. encryption and compression, compression 36. encryption and compression, encryption 37. encryption and compression, security 38. hardware and architecture, distributed architectures 39. hardware and architecture, high performance computing 40. hardware and architecture, input output and storage 41. hardware and architecture, logic design 42. hardware and architecture, memory structures 43. hardware and architecture, microprogramming 44. hardware and architecture, vlsi 45. human computer interaction, cooperative 46. human computer interaction, graphics and virtual reality 47. human computer interaction, interface design 48. human computer interaction, multimedia 49. human computer interaction, wearable computers 50. information retrieval, digital library 51. information retrieval, extraction 52. information retrieval, filtering 53. information retrieval, retrieval 54. nan 55. networking, internet 56. networking, protocols 57. networking, routing 58. networking, wireless 59. operating systems, distributed 60. operating systems, fault tolerance 61. operating systems, memory management 62. operating systems, realtime 63. programming, compiler design 64. programming, debugging 65. programming, functional 66. programming, garbage collection 67. programming, java 68. programming, logic 69. programming, object oriented 70. programming, semantics 71. programming, software development'
    new_string = '1. operating systems, memory management 2. artificial intelligence, planning 3. artificial intelligence, vision and pattern recognition 4. artificial intelligence, machine learning, case-based 5. artificial intelligence, agents 6. artificial intelligence, machine learning, probabilistic methods 7. operating systems, distributed 8. artificial intelligence, machine learning, genetic algorithms 9. human computer interaction, graphics and virtual reality 10. programming, object oriented 11. encryption and compression, encryption 12. networking, protocols 13.  programming, software development 14. programming, compiler design 15. artificial intelligence, machine learning, theory 16. artificial intelligence, machine learning, neural networks 17. programming, logic 18. operating systems, realtime 19. artificial intelligence, speech 20. artificial intelligence, robotics 21. artificial intelligence, games and search'
    new_string_none = '1. artificial intelligence, data mining 2. artificial intelligence, expert systems 3. artificial intelligence, knowledge representation 4. artificial intelligence, machine learning, reinforcement learning 5. artificial intelligence, machine learning, rule learning 6. artificial intelligence, nlp 7. artificial intelligence, theorem proving 8. data structures, algorithms and theory, computational complexity 9. data structures, algorithms and theory, computational geometry 10. data structures, algorithms and theory, formal languages 11. data structures, algorithms and theory, hashing 12. data structures, algorithms and theory, logic 13. data structures, algorithms and theory, parallel 14. data structures, algorithms and theory, quantum computing 15. data structures, algorithms and theory, randomized 16. data structures, algorithms and theory, sorting 17. databases, concurrency 18. databases, deductive 19. databases, object oriented 20. databases, performance 21. databases, query evaluation 22. databases, relational 23. databases, temporal 24. encryption and compression, compression 25. encryption and compression, security 26. hardware and architecture, distributed architectures 27. hardware and architecture, high performance computing 28. hardware and architecture, input output and storage 29. hardware and architecture, logic design 30. hardware and architecture, memory structures 31. hardware and architecture, microprogramming 32. hardware and architecture, vlsi 33. human computer interaction, cooperative 34. human computer interaction, interface design 35. human computer interaction, multimedia 36. human computer interaction, wearable computers 37. information retrieval, digital library 38. information retrieval, extraction 39. information retrieval, filtering 40. information retrieval, retrieval 41. nan 42. networking, internet 43. networking, routing 44. networking, wireless 45. operating systems, fault tolerance 46. programming, debugging 47. programming, functional 48. programming, garbage collection 49. programming, java 50. programming, semantics'
    prompt_file = load_file(args.prompting_file)
    print(f'total: {len(prompt_file)}')
    for idx, instruct_item in enumerate(prompt_file):
        qs = instruct_item["conversations"][0]["value"]
        ans = instruct_item['conversations'][1]['value']
        category = [cat for cat in categories if cat in ans]
        chosen = 0
        for cat in category:
            if cat in selected_categories:
                selected_indices.append(idx)
                prompt_file[idx]['conversations'][0]['value'] = qs.replace(old_string, new_string)
                chosen = 1
                break
        if chosen == 0:
            none_selected_indices.append(idx)
            prompt_file[idx]['conversations'][0]['value'] = qs.replace(old_string, new_string_none)
    
    # selected_items = [prompt_file[idx] for idx in selected_indices]
    # output_file = args.output_path + 'selected_items.json'
    # with open(output_file, 'w', encoding='utf-8') as file:
    #     json.dump(selected_items, file, ensure_ascii=False, indent=4)

    # print(f"选取的元素共{len(selected_indices)}个，已保存到文件 {output_file}")
    none_selected_items = [prompt_file[idx] for idx in none_selected_indices]
    output_file = args.output_path + 'none_selected_test_items.json'
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(none_selected_items, file, ensure_ascii=False, indent=4)
    
    print(f'OOD的元素共{len(none_selected_items)}个，已保存到文件 {output_file}')


def separate(args):
    train_set = []
    test_set = []
    category_counts = defaultdict(int)
    selected_items = load_file(args.selected_file)
    print(f"total: {len(selected_items)}")
    for idx, instruct_item in enumerate(selected_items):
        ans = instruct_item['conversations'][1]['value']
        category = [cat for cat in categories if cat in ans]
        for cat in category:
            category_counts[cat.strip()] += 1
            if category_counts[cat.strip()] <= 20:
                train_set.append(idx)
            elif category_counts[cat.strip()] > 50 and category_counts[cat.strip()] <= 60:
                test_set.append(idx)
            # else:
            #     train_set.append(idx)
    train_items = [selected_items[idx] for idx in train_set]
    test_items = [selected_items[idx] for idx in test_set]
    print(f'train: {len(train_items)}, test: {len(test_items)}.')
    train_file = args.output_path + 'train_items_{}_shots.json'.format(20)
    # test_file = args.output_path + 'test_items.json'
    with open(train_file, 'w', encoding='utf-8') as file:
        json.dump(train_items, file, ensure_ascii=False, indent=4)
    # with open(test_file, 'w', encoding='utf-8') as file:
    #     json.dump(test_items, file, ensure_ascii=False, indent=4)
    
    # clip_graph, args_graph= load_model_pretrained(CLIP, '/home/cjz/GraphGPT/clip_gt_arxiv')
    # graph_tower = graph_transformer(args_graph)
    # graph_tower_dict = torch.load('/home/cjz/GraphGPT/checkpoints/few-shot/tuned_graph_tower.bin')
    # new_keys = [k.replace('model.graph_tower.', '') for k in graph_tower_dict.keys()]
    # modified_dict = dict(zip(new_keys, graph_tower_dict.values()))
    # graph_tower.load_state_dict(modified_dict)
    
def answer_stacstics():
    answer_file = load_file('/home/cjz/GraphGPT/output_eva_cora/arxiv_test_res_0_210_with_rag.json')
    print(f'Total: {len(answer_file)}')
    mis_lead_and_wrong = 0
    mis_lead_but_right = 0
    right_lead_and_right = 0
    right_lead_but_wrong = 0
    for idx, answer_item in enumerate(answer_file):
        rag_categories = [cat for cat in selected_categories if cat in answer_item['rag label']]
        true_category = [cat for cat in selected_categories if cat in answer_item['label']]
        if true_category != []:
            true_category = true_category[0]
        else:
            true_category = None
        res_category = [cat for cat in selected_categories if cat in answer_item['res']]
        if res_category != []:
            res_category = res_category[0]
        else:
            res_category = None
        if true_category in rag_categories and res_category != true_category and res_category != None:
            right_lead_but_wrong += 1
        if true_category in rag_categories and res_category == true_category and res_category != None:
            right_lead_and_right += 1
        if true_category not in rag_categories and res_category != true_category and res_category != None:
            mis_lead_and_wrong += 1
        if true_category not in rag_categories and true_category == res_category and res_category != None:
            mis_lead_but_right += 1
    print(f'Mis_lead_and_wrong: {mis_lead_and_wrong}, mis_lead_but_right: {mis_lead_but_right}, right_lead_and_right: {right_lead_and_right} right_lead_but_wrong: {right_lead_but_wrong}')
    
def five_classification():
    test_items = load_file(args.test_file)
    train_items = load_file(args.train_file)
    test_set_01, test_set_02, test_set_03, test_set_04 = [], [], [], []
    category_counts = defaultdict(int)
    train_set_01_5shots, train_set_02_5shots, train_set_03_5shots, train_set_04_5shots = [], [], [], []
    train_set_01_20shots, train_set_02_20shots, train_set_03_20shots, train_set_04_20shots = [], [], [], []
    train_set_01_50shots, train_set_02_50shots, train_set_03_50shots, train_set_04_50shots = [], [], [], []
    train_set_5shots = [train_set_01_5shots, train_set_02_5shots, train_set_03_5shots, train_set_04_5shots]
    train_set_20shots = [train_set_01_20shots, train_set_02_20shots, train_set_03_20shots, train_set_04_20shots]
    train_set_50shots = [train_set_01_50shots, train_set_02_50shots, train_set_03_50shots, train_set_04_50shots]
    test_sets = [test_set_01, test_set_02, test_set_03, test_set_04]
    keys_list = list(five_ways_categories.keys())
    print(f'total: {len(test_items)}')
    old_string = '1. operating systems, memory management 2. artificial intelligence, planning 3. artificial intelligence, vision and pattern recognition 4. artificial intelligence, machine learning, case-based 5. artificial intelligence, agents 6. artificial intelligence, machine learning, probabilistic methods 7. operating systems, distributed 8. artificial intelligence, machine learning, genetic algorithms 9. human computer interaction, graphics and virtual reality 10. programming, object oriented 11. encryption and compression, encryption 12. networking, protocols 13.  programming, software development 14. programming, compiler design 15. artificial intelligence, machine learning, theory 16. artificial intelligence, machine learning, neural networks 17. programming, logic 18. operating systems, realtime 19. artificial intelligence, speech 20. artificial intelligence, robotics 21. artificial intelligence, games and search'
    new_strings = [
        '1. operating systems, memory management 2. artificial intelligence, planning 3. artificial intelligence, vision and pattern recognition 4. artificial intelligence, machine learning, case-based 5. artificial intelligence, agents',
        '1. artificial intelligence, machine learning, probabilistic methods 2. operating systems, distributed 3. artificial intelligence, machine learning, genetic algorithms 4. human computer interaction, graphics and virtual reality 5. programming, object oriented',
        '1. encryption and compression, encryption 2. networking, protocols 3.  programming, software development 4. programming, compiler design 5. artificial intelligence, machine learning, theory',
        '1. artificial intelligence, machine learning, neural networks 2. programming, logic 3. operating systems, realtime 4. artificial intelligence, robotics 5. artificial intelligence, games and search'
    ]
    # for idx, instruct_item in enumerate(test_items):
    #     qs = instruct_item['conversations'][0]['value']
    #     ans = instruct_item['conversations'][1]['value']
    #     try:
    #         index = keys_list.index(ans)
    #         test_set_index = index // 5
    #         instruct_item['conversations'][0]['value'] = qs.replace(old_string, new_strings[test_set_index])
    #         test_sets[test_set_index].append(instruct_item)
    #     except ValueError:
    #         print(f"'{ans}' not found in the selected categories.")
    for idx, instruct_item in enumerate(train_items):
        qs = instruct_item['conversations'][0]['value']
        ans = instruct_item['conversations'][1]['value']
        try:
            index = keys_list.index(ans)
            train_set_index = index // 5
            category_counts[ans] += 1
            instruct_item['conversations'][0]['value'] = qs.replace(old_string, new_strings[train_set_index])
            if category_counts[ans] <= 5:                
                train_set_5shots[train_set_index].append(instruct_item)
            if category_counts[ans] <= 20:
                train_set_20shots[train_set_index].append(instruct_item)
            if category_counts[ans] <= 50:
                train_set_50shots[train_set_index].append(instruct_item)
        except ValueError:
            print(f"'{ans}' not found in the selected categories.")
    # for i, test_set in enumerate([test_set_01, test_set_02, test_set_03, test_set_04], start=1):
    #     with open(f'/home/cjz/SFTonGFM/reshape/test_items_5ways_0{i}.json', 'w', encoding='utf-8') as file:
    #         json.dump(test_set, file, ensure_ascii=False, indent=4)
    for i, train_set in enumerate(train_set_5shots, start=1):
        with open(f'/home/cjz/SFTonGFM/data/cora/train_5shotss_5ways_0{i}.json', 'w', encoding='utf-8') as file:
            json.dump(train_set, file, ensure_ascii=False, indent=4)
    for i, train_set in enumerate(train_set_20shots, start=1):
        with open(f'/home/cjz/SFTonGFM/data/cora/train_20shots_5ways_0{i}.json', 'w', encoding='utf-8') as file:
            json.dump(train_set, file, ensure_ascii=False, indent=4)
    for i, train_set in enumerate(train_set_50shots, start=1):
        with open(f'/home/cjz/SFTonGFM/data/cora/train_50shots_5ways_0{i}.json', 'w', encoding='utf-8') as file:
            json.dump(train_set, file, ensure_ascii=False, indent=4)

def G2P2_arxiv():
    tit_list = []
    while len(tit_list) < 169343 :
        tit_list.append(None)
    mis_match = 0
    arxiv_items = load_file('/home/cjz/GraphGPT/stage_2_instruct/arxiv_pub_node_st_cot_link_mix.json')
    graph_data = torch.load('/home/cjz/GraphGPT/graph_data/graph_data_all.pt')
    for instruct_item in enumerate(arxiv_items):
        idx = instruct_item[1]['id']
        if 'arxiv' in idx:
            id = instruct_item[1]['graph']['node_idx']
            match = re.search(r'\nAbstract:(.*?)\n Question:', instruct_item[1]['conversations'][0]['value'], re.DOTALL)
            if match:
                content_between = match.group(1).strip()
                tit_list[id] = content_between
            else:
                mis_match += 1
    print(mis_match)
    with open('/home/cjz/G2P2/data/arxiv_tit_list.json', 'w', encoding='utf-8') as file:
        json.dump(tit_list, file, ensure_ascii=False, indent=4)
                
def tit_gen_data():
    tit_list = []
    abs_list = []
    train_tit_items = []
    test_tit_items = []
    graph_data = torch.load('/home/cjz/GraphGPT/graph_data/graph_data_all.pt')
    # train = load_file(args.train_file)
    # test = load_file(args.test_file)
    cora_instruct = load_file(args.prompting_file)
    # cora_instruct = train + test
    split = int(len(cora_instruct) * 0.8)
    train_items = cora_instruct[:split]
    test_items = cora_instruct[split:]
    with open('/home/cjz/G2P2/data/train_text.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            tit_list.append(line[1])
            abs_list.append(line[2])
    for idx, instrcut_item in enumerate(train_items):
        node_id = instrcut_item['graph']['node_idx']
        qs = instrcut_item['conversations'][0]['value']
        old_string = 'Which of the following subcategories of computer science does this paper belong to: 1. artificial intelligence, agents 2. artificial intelligence, data mining 3. artificial intelligence, expert systems 4. artificial intelligence, games and search 5. artificial intelligence, knowledge representation 6. artificial intelligence, machine learning, case-based 7. artificial intelligence, machine learning, genetic algorithms 8. artificial intelligence, machine learning, neural networks 9. artificial intelligence, machine learning, probabilistic methods 10. artificial intelligence, machine learning, reinforcement learning 11. artificial intelligence, machine learning, rule learning 12. artificial intelligence, machine learning, theory 13. artificial intelligence, nlp 14. artificial intelligence, planning 15. artificial intelligence, robotics 16. artificial intelligence, speech 17. artificial intelligence, theorem proving 18. artificial intelligence, vision and pattern recognition 19. data structures  algorithms and theory, computational complexity 20. data structures  algorithms and theory, computational geometry 21. data structures  algorithms and theory, formal languages 22. data structures  algorithms and theory, hashing 23. data structures  algorithms and theory, logic 24. data structures  algorithms and theory, parallel 25. data structures  algorithms and theory, quantum computing 26. data structures  algorithms and theory, randomized 27. data structures  algorithms and theory, sorting 28. databases, concurrency 29. databases, deductive 30. databases, object oriented 31. databases, performance 32. databases, query evaluation 33. databases, relational 34. databases, temporal 35. encryption and compression, compression 36. encryption and compression, encryption 37. encryption and compression, security 38. hardware and architecture, distributed architectures 39. hardware and architecture, high performance computing 40. hardware and architecture, input output and storage 41. hardware and architecture, logic design 42. hardware and architecture, memory structures 43. hardware and architecture, microprogramming 44. hardware and architecture, vlsi 45. human computer interaction, cooperative 46. human computer interaction, graphics and virtual reality 47. human computer interaction, interface design 48. human computer interaction, multimedia 49. human computer interaction, wearable computers 50. information retrieval, digital library 51. information retrieval, extraction 52. information retrieval, filtering 53. information retrieval, retrieval 54. nan 55. networking, internet 56. networking, protocols 57. networking, routing 58. networking, wireless 59. operating systems, distributed 60. operating systems, fault tolerance 61. operating systems, memory management 62. operating systems, realtime 63. programming, compiler design 64. programming, debugging 65. programming, functional 66. programming, garbage collection 67. programming, java 68. programming, logic 69. programming, object oriented 70. programming, semantics 71. programming, software development ? Directly give the full name of the most likely category of this paper. '
        new_string = 'Please generate a suitable title for this paper. Directly give the title.'
        match = re.search(r'\nAbstract:(.*?)\n Question:', instrcut_item['conversations'][0]['value'], re.DOTALL)
        if match:
            qs = qs.replace(match.group(1), abs_list[node_id])
        instrcut_item['conversations'][0]['value'] = qs.replace(old_string, new_string)
        instrcut_item['conversations'][1]['value'] = tit_list[node_id]
        train_tit_items.append(instrcut_item)
    for idx, instrcut_item in enumerate(test_items):
        node_id = instrcut_item['graph']['node_idx']
        qs = instrcut_item['conversations'][0]['value']
        old_string = 'Which of the following subcategories of computer science does this paper belong to: 1. artificial intelligence, agents 2. artificial intelligence, data mining 3. artificial intelligence, expert systems 4. artificial intelligence, games and search 5. artificial intelligence, knowledge representation 6. artificial intelligence, machine learning, case-based 7. artificial intelligence, machine learning, genetic algorithms 8. artificial intelligence, machine learning, neural networks 9. artificial intelligence, machine learning, probabilistic methods 10. artificial intelligence, machine learning, reinforcement learning 11. artificial intelligence, machine learning, rule learning 12. artificial intelligence, machine learning, theory 13. artificial intelligence, nlp 14. artificial intelligence, planning 15. artificial intelligence, robotics 16. artificial intelligence, speech 17. artificial intelligence, theorem proving 18. artificial intelligence, vision and pattern recognition 19. data structures  algorithms and theory, computational complexity 20. data structures  algorithms and theory, computational geometry 21. data structures  algorithms and theory, formal languages 22. data structures  algorithms and theory, hashing 23. data structures  algorithms and theory, logic 24. data structures  algorithms and theory, parallel 25. data structures  algorithms and theory, quantum computing 26. data structures  algorithms and theory, randomized 27. data structures  algorithms and theory, sorting 28. databases, concurrency 29. databases, deductive 30. databases, object oriented 31. databases, performance 32. databases, query evaluation 33. databases, relational 34. databases, temporal 35. encryption and compression, compression 36. encryption and compression, encryption 37. encryption and compression, security 38. hardware and architecture, distributed architectures 39. hardware and architecture, high performance computing 40. hardware and architecture, input output and storage 41. hardware and architecture, logic design 42. hardware and architecture, memory structures 43. hardware and architecture, microprogramming 44. hardware and architecture, vlsi 45. human computer interaction, cooperative 46. human computer interaction, graphics and virtual reality 47. human computer interaction, interface design 48. human computer interaction, multimedia 49. human computer interaction, wearable computers 50. information retrieval, digital library 51. information retrieval, extraction 52. information retrieval, filtering 53. information retrieval, retrieval 54. nan 55. networking, internet 56. networking, protocols 57. networking, routing 58. networking, wireless 59. operating systems, distributed 60. operating systems, fault tolerance 61. operating systems, memory management 62. operating systems, realtime 63. programming, compiler design 64. programming, debugging 65. programming, functional 66. programming, garbage collection 67. programming, java 68. programming, logic 69. programming, object oriented 70. programming, semantics 71. programming, software development ? Directly give the full name of the most likely category of this paper. '
        new_string = 'Please generate a suitable title for this paper. Directly give the title.'
        match = re.search(r'\nAbstract:(.*?)\n Question:', instrcut_item['conversations'][0]['value'], re.DOTALL)
        if match:
            qs = qs.replace(match.group(1), abs_list[node_id])
        instrcut_item['conversations'][0]['value'] = qs.replace(old_string, new_string)
        instrcut_item['conversations'][1]['value'] = tit_list[node_id]
        test_tit_items.append(instrcut_item)    
    # train_items = load_file('/home/cjz/SFTonGFM/reshape/train_items.json')
    train_file = args.output_path + 'train_items_tit_gen.json'
    test_file = args.output_path + 'test_items_tit_gen.json'
    train_file_5shots = args.output_path + 'train_items_tit_gen_5shots.json'
    test_file_5shots = args.output_path + 'test_items_tit_gen_5shots.json'
    
    all_items = train_tit_items + test_tit_items
    split = int(len(all_items) * 0.33)
    train_tit_items_5shots = all_items[:split]
    test_tit_items_5shots = all_items[split:]
    with open(train_file, 'w', encoding='utf-8') as file:
        json.dump(train_tit_items, file, ensure_ascii=False, indent=4)
    with open(test_file, 'w', encoding='utf-8') as file:
        json.dump(test_tit_items, file, ensure_ascii=False, indent=4)
    with open(train_file_5shots, 'w', encoding='utf-8') as file:
        json.dump(train_tit_items_5shots, file, ensure_ascii=False, indent=4)
    with open(test_file_5shots, 'w', encoding='utf-8') as file:
        json.dump(test_tit_items_5shots, file, ensure_ascii=False, indent=4)

def task_text():
    clas_71ways_text = 'Which of the following subcategories of computer science does this paper belong to: 1. artificial intelligence, agents 2. artificial intelligence, data mining 3. artificial intelligence, expert systems 4. artificial intelligence, games and search 5. artificial intelligence, knowledge representation 6. artificial intelligence, machine learning, case-based 7. artificial intelligence, machine learning, genetic algorithms 8. artificial intelligence, machine learning, neural networks 9. artificial intelligence, machine learning, probabilistic methods 10. artificial intelligence, machine learning, reinforcement learning 11. artificial intelligence, machine learning, rule learning 12. artificial intelligence, machine learning, theory 13. artificial intelligence, nlp 14. artificial intelligence, planning 15. artificial intelligence, robotics 16. artificial intelligence, speech 17. artificial intelligence, theorem proving 18. artificial intelligence, vision and pattern recognition 19. data structures  algorithms and theory, computational complexity 20. data structures  algorithms and theory, computational geometry 21. data structures  algorithms and theory, formal languages 22. data structures  algorithms and theory, hashing 23. data structures  algorithms and theory, logic 24. data structures  algorithms and theory, parallel 25. data structures  algorithms and theory, quantum computing 26. data structures  algorithms and theory, randomized 27. data structures  algorithms and theory, sorting 28. databases, concurrency 29. databases, deductive 30. databases, object oriented 31. databases, performance 32. databases, query evaluation 33. databases, relational 34. databases, temporal 35. encryption and compression, compression 36. encryption and compression, encryption 37. encryption and compression, security 38. hardware and architecture, distributed architectures 39. hardware and architecture, high performance computing 40. hardware and architecture, input output and storage 41. hardware and architecture, logic design 42. hardware and architecture, memory structures 43. hardware and architecture, microprogramming 44. hardware and architecture, vlsi 45. human computer interaction, cooperative 46. human computer interaction, graphics and virtual reality 47. human computer interaction, interface design 48. human computer interaction, multimedia 49. human computer interaction, wearable computers 50. information retrieval, digital library 51. information retrieval, extraction 52. information retrieval, filtering 53. information retrieval, retrieval 54. nan 55. networking, internet 56. networking, protocols 57. networking, routing 58. networking, wireless 59. operating systems, distributed 60. operating systems, fault tolerance 61. operating systems, memory management 62. operating systems, realtime 63. programming, compiler design 64. programming, debugging 65. programming, functional 66. programming, garbage collection 67. programming, java 68. programming, logic 69. programming, object oriented 70. programming, semantics 71. programming, software development ? Directly give the full name of the most likely category of this paper.'
    clas_21ways_text = 'Which of the following subcategories of computer science does this paper belong to: 1. operating systems, memory management 2. artificial intelligence, planning 3. artificial intelligence, vision and pattern recognition 4. artificial intelligence, machine learning, case-based 5. artificial intelligence, agents 6. artificial intelligence, machine learning, probabilistic methods 7. operating systems, distributed 8. artificial intelligence, machine learning, genetic algorithms 9. human computer interaction, graphics and virtual reality 10. programming, object oriented 11. encryption and compression, encryption 12. networking, protocols 13.  programming, software development 14. programming, compiler design 15. artificial intelligence, machine learning, theory 16. artificial intelligence, machine learning, neural networks 17. programming, logic 18. operating systems, realtime 19. artificial intelligence, speech 20. artificial intelligence, robotics 21. artificial intelligence, games and search ? Directly give the full name of the most likely category of this paper.'
    clas_5ways_01_text = 'Which of the following subcategories of computer science does this paper belong to: 1. operating systems, memory management 2. artificial intelligence, planning 3. artificial intelligence, vision and pattern recognition 4. artificial intelligence, machine learning, case-based 5. artificial intelligence, agents ? Directly give the full name of the most likely category of this paper.'
    clas_5ways_02_text = 'Which of the following subcategories of computer science does this paper belong to: 1. artificial intelligence, machine learning, probabilistic methods 2. operating systems, distributed 3. artificial intelligence, machine learning, genetic algorithms 4. human computer interaction, graphics and virtual reality 5. programming, object oriented ? Directly give the full name of the most likely category of this paper.'
    clas_5ways_03_text = 'Which of the following subcategories of computer science does this paper belong to: 1. encryption and compression, encryption 2. networking, protocols 3.  programming, software development 4. programming, compiler design 5. artificial intelligence, machine learning, theory ? Directly give the full name of the most likely category of this paper.'
    clas_5ways_04_text = 'Which of the following subcategories of computer science does this paper belong to: 1. artificial intelligence, machine learning, neural networks 2. programming, logic 3. operating systems, realtime 4. artificial intelligence, robotics 5. artificial intelligence, games and search ? Directly give the full name of the most likely category of this paper.'
    tit_gen_text = 'Please generate a suitable title for this paper. Directly give the title.'
    graph_match_text = 'please reorder the list of papers according to the order of graph tokens (i.e., complete the matching of graph tokens and papers).'
    arxiv_clas_text = 'Which arXiv CS sub-category does this paper belong to? Give the most likely arXiv CS sub-categories of this paper directly, in the form \"cs.XX\" with full name of the category.'
    task_text_dict ={
        'cora_71ways': clas_71ways_text,
        'cora_21ways': clas_21ways_text,
        'cora_5ways_1': clas_5ways_01_text,
        'cora_5ways_2': clas_5ways_02_text,
        'cora_5ways_3': clas_5ways_03_text,
        'cora_5ways_4': clas_5ways_04_text,
        'cora_tit_gen': tit_gen_text,
        'arxiv_graph_match': graph_match_text,
        'arxiv_clas': arxiv_clas_text,
    }
    with open('./task_text.json', 'w', encoding='utf-8') as file:
        json.dump(task_text_dict, file, ensure_ascii=False, indent=4)
    
def seperate_arxiv_pubmed():
    all_items = load_file('/home/cjz/GraphGPT/stage_2_instruct/arxiv_pub_node_st_cot_link_mix.json')
    arxiv_node, pub_node, pub_link = [], [], []
    for idx, item in enumerate(all_items):
        if 'arxiv' in item['id']:
            arxiv_node.append(item)
        elif 'pubmed' in item['id']:
            if 'LP' in item['id']:
                pub_link.append(item)
            else:
                pub_node.append(item)
    with open('./arxiv_node_std.json', 'w', encoding='utf-8') as file:
        json.dump(arxiv_node, file, ensure_ascii=False, indent=4)
    with open('./pub_link_std.json', 'w', encoding='utf-8') as file:
        json.dump(pub_link, file, ensure_ascii=False, indent=4)
    # with open('./pub_node_std.json', 'w', encoding='utf-8') as file:
    #     json.dump(pub_node, file, ensure_ascii=False, indent=4)
    print('done')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompting_file", type=str, default='/home/cjz/GraphGPT/evaluation/cora_test_instruct_std.json')
    # parser.add_argument("--graph_data_path", type=str, default=None)
    parser.add_argument("--selected_file", type=str, default='/home/cjz/GraphGPT/reshape/selected_items.json')
    parser.add_argument("--output_path", type=str, default='/home/cjz/SFTonGFM/reshape/')
    parser.add_argument("--shot_num", type=int, default=50)
    parser.add_argument("--test_file", type=str, default='/home/cjz/SFTonGFM/data/cora/test_items.json')
    parser.add_argument("--train_file", type=str, default='/home/cjz/SFTonGFM/data/cora/train_items_50_shots.json')

    args = parser.parse_args()
    
    # reshape(args)
    # separate(args)
    # answer_stacstics()
    five_classification()
    # G2P2_arxiv()
    # tit_gen_data()
    # seperate_arxiv_pubmed()