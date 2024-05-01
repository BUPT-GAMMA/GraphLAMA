import random
import torch

# 设置随机数的范围
lower_bound = 34000
upper_bound = 34500

bias_bound_low = 1.5
bias_bound_up = 2.5

# 生成一个随机数并保留两位小数
random_number = round(random.uniform(lower_bound, upper_bound), 2)
random_bias = round(random.uniform(bias_bound_low, bias_bound_up), 2)

print(f'{random_number}±{random_bias}')