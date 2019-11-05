import os,sys
import random
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from sklearn import preprocessing
import json

class Data_processer():
    def __init__(self, args):
        self.args = args
        self.mode = args.mode
        self.data_path = args.data_path
        self.n_fold = args.n_fold

    # convert each char of words to its index in emb_dict
    def convert_char_to_index(self, emb_dict, words):
        new_words = []
        if isinstance(words, list):
            pass
        elif isinstance(words, str):
            words = words.split(' ')
        for word in words:
            if word in emb_dict:
                new_words.append(emb_dict[word])
            else:
                new_words.append(emb_dict['UNK'])
        #print ('words', words)
        #print ('new_words',new_words)
        return new_words
    
    #pad word according to max_len
    def pad_words(self, words_idx, emb_dict, max_len=-1):
        pad_idx = emb_dict['PAD']
        #print ('pad_debug:', words_idx, len(words_idx), max_len)
        words_idx = words_idx + [pad_idx]*(max_len-len(words_idx)) 
        #print ('pad_debug_after:', words_idx, len(words_idx), max_len)
        return words_idx
        

    def data_preprocesser(self):
        file_list = os.listdir(self.data_path)
        file_list.sort()
        fold_data_dict = defaultdict(list)
        if self.mode[:-1] == 'char':
            n_gram = int(self.mode[-1])
            vectorizer = CountVectorizer(analyzer='char', ngram_range=(n_gram,n_gram))
            analyzer = vectorizer.build_analyzer()
        emb_list = set([])
        data_len = 0
        max_sen_len = 0
        user_idx = 0

        for file in file_list:
            file_path = self.data_path + file
            with open(file_path, 'r') as f:
                line_idx = 0
                for line in f:
                    #print(line)
                    line = line.strip('\n').lower()
                    line = line.split('\t')
                    if len(line) != 2:
                        print(line)
                    if self.mode == 'word':
                        line[0] = line[0].split(' ')
                    else:
                        line[0] = analyzer(line[0])
                    #print(bigram_text)
                    for elem in line[0]:
                        if elem not in emb_list:
                            emb_list.add(elem)
#                         if len(bigram_text) > 140:
#                             continue
                    if len(line[0]) > max_sen_len:
                        max_sen_len = len(line[0])
                    text_len = len(line[0])
                    i = int(line_idx/(1000/self.n_fold))
                    fold_data_dict['fold%d_text'%i].append(line[0])
                    fold_data_dict['fold%d_user'%i].append(user_idx)
                    fold_data_dict['fold%d_len'%i].append(text_len)
                    data_len += 1
                    line_idx += 1
            user_idx += 1
        print('max_sen_len:', max_sen_len)
        emb_list = list(emb_list)
        emb_list.sort()
        emb_list += ['UNK', 'PAD']
        emb_num = len(emb_list)
        emb_dict = dict([(e, idx) for idx, e in enumerate(emb_list)])
        print('emb_list_len', len(emb_list))
        #convert text to id in emb_dict
        fold_id_dict = defaultdict(list)
        pbar = tqdm(total=data_len)
        for i in range(self.n_fold):
            text_list = fold_data_dict['fold%d_text'%i]
            for text in text_list:
                text_id = self.convert_char_to_index(emb_dict, text)
                text_id = self.pad_words(text_id, emb_dict, max_sen_len)
                fold_id_dict['fold%d_text'%i].append(text_id)
                pbar.update()
        pbar.close()
        data_info = {'emb_list':emb_list, 'emb_dict':emb_dict, 'fold_id_dict':fold_id_dict, 'fold_data_dict':fold_data_dict,\
                    'emb_num':emb_num}
        
        return data_info

class Dataloader():
    def __init__(self, args, data_info, fold_num):
        self.args = args
        self.n_fold = args.n_fold
        self.batch_size = args.batch_size
        self.data_info = data_info
        self.fold_num = fold_num
        self.emb_list = data_info['emb_list']
        self.emb_dict = data_info['emb_dict']
        self.fold_id_dict = data_info['fold_id_dict']
        self.fold_data_dict = data_info['fold_data_dict']
        
        
    def data_spliter(self):
        train_text = []
        train_len = []
        train_user = []
        test_text = []
        test_len = []
        test_user = []
        for i in range(self.n_fold):
            if i == self.fold_num:
                test_text += self.fold_id_dict['fold%d_text'%i]
                test_user += self.fold_data_dict['fold%d_user'%i]
                test_len += self.fold_data_dict['fold%d_len'%i]
            else:
                train_text += self.fold_id_dict['fold%d_text'%i]
                train_user += self.fold_data_dict['fold%d_user'%i]
                train_len += self.fold_data_dict['fold%d_len'%i]
        train_label = train_user
        test_label = test_user
        return train_text, test_text, train_label, test_label, train_len, test_len
    
    def data_iter_train(self):
        batch_size=self.batch_size
        train_text, test_text, train_label, test_label, train_len, test_len = self.data_spliter()
        data_temp = list(zip(train_text, train_label, train_len))
        #random.seed(10)
        random.shuffle(data_temp)
        train_text, train_label, train_len = zip(*data_temp)
        #train_text, train_label = np.array(train_text), np.array(train_label)
        batch_num = int(len(train_text)/batch_size)
        if batch_num*batch_size < len(train_text):
            batch_num += 1
        
        for batch_i in range(batch_num):
            batch_start = batch_i * batch_size
            batch_end = min((batch_i+1)*batch_size, len(train_text))
            
            batch_train_text = train_text[batch_start: batch_end]
            batch_train_label = train_label[batch_start: batch_end]
            batch_train_len = train_len[batch_start: batch_end]
            
            #sort
            batch_train_text = [x for (y,x) in sorted(zip(batch_train_len,batch_train_text), reverse=True)]
            batch_train_label = [x for (y,x) in sorted(zip(batch_train_len,batch_train_label), reverse=True)]
            batch_train_len = sorted(batch_train_len, reverse= True)
            
            batch_elem = {'batch_train_text':batch_train_text, 'batch_train_label': batch_train_label, 'batch_size': batch_size,\
                         'total_len':len(train_text), 'batch_train_len': batch_train_len}
        
            yield batch_elem
        
