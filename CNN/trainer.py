from data_loader import Data_processer
from data_loader import Dataloader
from config import Argparse
from cnn_n import CNN_n
import numpy as np
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import torch.optim as optim
from torch.autograd import Variable

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class Trainer():
    def __init__(self, data_loader, cnn_n, args, fold_num):
        self.data_loader = data_loader
        self.cnn_n = cnn_n
        self.args = args
        self.epoch = args.epoch
        self.optim = optim.Adam(self.cnn_n.parameters(), lr=self.args.lr)
        self.checkpoint_path = args.checkpoint_path
        self.fold_num = fold_num
        
    def test(self, load_para, batch_size=512):
        _, test_text, _, test_label = self.data_loader.data_spliter()
        #load parameters
        if load_para:
            checkpoint = self.checkpoint_path + 'fold%d/m-best.pth.tar'%self.fold_num
            if checkpoint != None:
                model_CKPT = torch.load(checkpoint)
                #print('loading checkpoint...')
                self.cnn_n.load_state_dict(model_CKPT['state_dict'])
                #print('checkpoint load done!')
        self.cnn_n.eval()
        batch_num = int(len(test_text)/batch_size)
        correct = 0
        
        for batch_i in range(batch_num+1):
            batch_start = batch_i * batch_size
            batch_end = min((batch_i+1)*batch_size, len(test_text))
            
            batch_test_text = test_text[batch_start: batch_end]
            batch_test_label = test_label[batch_start: batch_end]
            
            batch_test_text_tensor = torch.LongTensor(batch_test_text).cuda()
            batch_test_label_tensor = torch.LongTensor(batch_test_label).cuda()
            
            predict = self.cnn_n(batch_test_text_tensor)
            predict_class = torch.argmax(F.softmax(predict, dim=1), dim=1).cpu().numpy()
            correct += (predict_class == batch_test_label).sum()
            
        return correct/(len(test_label))
        
        
        
    def train(self):
        loss = nn.CrossEntropyLoss()
        best_acc = 0
        for epoch in range(1, self.epoch+1):
            start = time.time()
            _, _, train_label, test_label = self.data_loader.data_spliter()
            train_generator = self.data_loader.data_iter_train()
            epoch_loss = 0.0
            epoch_loss_list = []
            correct = 0
            total_len = len(train_label)
            test_len = len(test_label)
            print('total_len:', total_len, 'test_len:', test_len)
            #pbar = tqdm(total=int(total_len/self.args.batch_size)+1)
            for batch_elem in train_generator:
                batch_train_text = batch_elem['batch_train_text']
                batch_train_label = batch_elem['batch_train_label']
                batch_size = batch_elem['batch_size']
                #pbar.update(1)
                
                self.optim.zero_grad()
                batch_loss = 0.0
                
                batch_train_text_tensor = torch.LongTensor(batch_train_text).cuda()
                batch_train_label_tensor = torch.LongTensor(batch_train_label).cuda()
                
                predict = self.cnn_n(batch_train_text_tensor)
                predict_class = torch.argmax(F.softmax(predict, dim=1), dim=1).cpu().numpy()
                correct += (predict_class == batch_train_label).sum()
                
                #batch_train_label_tensor = batch_train_label_tensor.unsqueeze(0).unsqueeze(0)
                batch_loss = loss(predict, batch_train_label_tensor)
                batch_loss.backward()
                self.optim.step()
                
                epoch_loss += batch_loss.item()
            
            #pbar.close()
            train_acc = correct/total_len
            test_acc = self.test(False, 512)
            epoch_loss_list.append(epoch_loss)
            end = time.time()
            print('epoch:', epoch, 'epoch_loss:', epoch_loss, 'train_acc:', train_acc, 'test_acc',test_acc, 'time:%fh'%((end-start)/3600))
            if best_acc < test_acc:
                best_acc = test_acc
                print('Checkpoint update!')
                if not os.path.exists(self.checkpoint_path):
                    os.mkdir(self.checkpoint_path)
                if not os.path.exists(self.checkpoint_path+'fold%d/'%self.fold_num):
                    os.mkdir(self.checkpoint_path+'fold%d/'%self.fold_num)
                lossMIN = min(epoch_loss_list)
                torch.save({'epoch': epoch, 'state_dict': self.cnn_n.state_dict(), 'best_loss': lossMIN,
                                'optimizer': self.optim.state_dict()},
                               self.checkpoint_path + 'fold%d/'%self.fold_num +'m-best.pth.tar')
        test_acc = self.test(True, 512)
        print('fold%d test accuracy:'%self.fold_num, test_acc)
        return test_acc

def main():
    args = Argparse()
    data = Data_processer(args)
    data_info = data.data_preprocesser()
    test_acc_list = []
    for i in range(1, 2):
        data_loader = Dataloader(args, data_info, i)
        cnn_n = CNN_n(args, data_info).cuda()
        doer = Trainer(data_loader, cnn_n, args, i)
        test_acc = doer.train()
        test_acc_list.append(test_acc)
    print('test accuracy:', sum(test_acc_list)/len(test_acc_list))

if __name__ == '__main__':
    main()
