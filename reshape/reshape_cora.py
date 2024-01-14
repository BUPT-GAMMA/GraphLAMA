import argparse
from tqdm import tqdm
import json
import os.path as osp
from collections import defaultdict
from graphgpt.model import *
import torch
import os

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

def load_file(file_path): 
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def reshape(args):
    selected_indices = []
    old_string = '1. artificial intelligence, agents 2. artificial intelligence, data mining 3. artificial intelligence, expert systems 4. artificial intelligence, games and search 5. artificial intelligence, knowledge representation 6. artificial intelligence, machine learning, case-based 7. artificial intelligence, machine learning, genetic algorithms 8. artificial intelligence, machine learning, neural networks 9. artificial intelligence, machine learning, probabilistic methods 10. artificial intelligence, machine learning, reinforcement learning 11. artificial intelligence, machine learning, rule learning 12. artificial intelligence, machine learning, theory 13. artificial intelligence, nlp 14. artificial intelligence, planning 15. artificial intelligence, robotics 16. artificial intelligence, speech 17. artificial intelligence, theorem proving 18. artificial intelligence, vision and pattern recognition 19. data structures  algorithms and theory, computational complexity 20. data structures  algorithms and theory, computational geometry 21. data structures  algorithms and theory, formal languages 22. data structures  algorithms and theory, hashing 23. data structures  algorithms and theory, logic 24. data structures  algorithms and theory, parallel 25. data structures  algorithms and theory, quantum computing 26. data structures  algorithms and theory, randomized 27. data structures  algorithms and theory, sorting 28. databases, concurrency 29. databases, deductive 30. databases, object oriented 31. databases, performance 32. databases, query evaluation 33. databases, relational 34. databases, temporal 35. encryption and compression, compression 36. encryption and compression, encryption 37. encryption and compression, security 38. hardware and architecture, distributed architectures 39. hardware and architecture, high performance computing 40. hardware and architecture, input output and storage 41. hardware and architecture, logic design 42. hardware and architecture, memory structures 43. hardware and architecture, microprogramming 44. hardware and architecture, vlsi 45. human computer interaction, cooperative 46. human computer interaction, graphics and virtual reality 47. human computer interaction, interface design 48. human computer interaction, multimedia 49. human computer interaction, wearable computers 50. information retrieval, digital library 51. information retrieval, extraction 52. information retrieval, filtering 53. information retrieval, retrieval 54. nan 55. networking, internet 56. networking, protocols 57. networking, routing 58. networking, wireless 59. operating systems, distributed 60. operating systems, fault tolerance 61. operating systems, memory management 62. operating systems, realtime 63. programming, compiler design 64. programming, debugging 65. programming, functional 66. programming, garbage collection 67. programming, java 68. programming, logic 69. programming, object oriented 70. programming, semantics 71. programming, software development'
    new_string = '1. operating systems, memory management 2. artificial intelligence, planning 3. artificial intelligence, vision and pattern recognition 4. artificial intelligence, machine learning, case-based 5. artificial intelligence, agents 6. artificial intelligence, machine learning, probabilistic methods 7. operating systems, distributed 8. artificial intelligence, machine learning, genetic algorithms 9. human computer interaction, graphics and virtual reality 10. programming, object oriented 11. encryption and compression, encryption 12. networking, protocols 13.  programming, software development 14. programming, compiler design 15. artificial intelligence, machine learning, theory 16. artificial intelligence, machine learning, neural networks 17. programming, logic 18. operating systems, realtime 19. artificial intelligence, speech 20. artificial intelligence, robotics 21. artificial intelligence, games and search'
    prompt_file = load_file(args.prompting_file)
    print(f'total: {len(prompt_file)}')
    for idx, instruct_item in enumerate(prompt_file):
        qs = instruct_item["conversations"][0]["value"]
        ans = instruct_item['conversations'][1]['value']
        category = [cat for cat in categories if cat in ans]
        for cat in category:
            if cat in selected_categories:
                selected_indices.append(idx)
                prompt_file[idx]['conversations'][0]['value'] = qs.replace(old_string, new_string)
                break
    
    selected_items = [prompt_file[idx] for idx in selected_indices]
    output_file = args.output_path + 'selected_items.json'
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(selected_items, file, ensure_ascii=False, indent=4)

    print(f"选取的元素共{len(selected_indices)}个，已保存到文件 {output_file}")

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
            if category_counts[cat.strip()] <= 50:
                train_set.append(idx)
            elif category_counts[cat.strip()] > 50 and category_counts[cat.strip()] <= 60:
                test_set.append(idx)
    train_items = [selected_items[idx] for idx in train_set]
    test_items = [selected_items[idx] for idx in test_set]
    print(f'train: {len(train_items)}, test: {len(test_items)}.')
    train_file = args.output_path + 'train_items.json'
    test_file = args.output_path + 'test_items.json'
    with open(train_file, 'w', encoding='utf-8') as file:
        json.dump(train_items, file, ensure_ascii=False, indent=4)
    with open(test_file, 'w', encoding='utf-8') as file:
        json.dump(test_items, file, ensure_ascii=False, indent=4)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompting_file", type=str, default='/home/cjz/GraphGPT/evaluation/cora_test_instruct_std.json')
    # parser.add_argument("--graph_data_path", type=str, default=None)
    parser.add_argument("--selected_file", type=str, default='/home/cjz/GraphGPT/reshape/selected_items.json')
    parser.add_argument("--output_path", type=str, default='/home/cjz/GraphGPT/reshape/')

    args = parser.parse_args()
    
    # reshape(args)
    # separate(args)
    answer_stacstics()
    