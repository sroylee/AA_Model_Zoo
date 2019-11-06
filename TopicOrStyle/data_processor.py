import os, sys
import random
import numpy as np
import re
import csv
from tqdm import tqdm

post_num = 100
user_num = 50
path = "/home/xiaojinhui/hzq/AA_Model_Zoo/datasets/twitter/%d_user_%d_posts/"%(user_num, post_num)
save_path = "%d_user_%d_posts_twitter/"%(user_num, post_num)
if not os.path.exists(save_path):
    os.mkdir(save_path)
dirs = os.listdir(path)

train_data = [["article", "class"]]
test_data = [["article", "class"]]
for dir in dirs:
    with open(path + dir, 'r') as f:
        i = 0
        for line in f:
            line = line.strip('\n')
            line = line.split('\t')
            #print(line[0])
            #line[0] = ''.join(line[0].split(' '))
            if len(line[0]) == 0:
                print(line)
            if i < post_num:
                train_data.append(line)
            else:
                test_data.append(line)
            i += 1


with open(save_path + 'test.csv' , 'w') as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL, dialect='excel')
            writer.writerows(test_data)
with open(save_path + 'train.csv', 'w') as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL, dialect='excel')
            writer.writerows(train_data)
