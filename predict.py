#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import torch
import pickle
from torch.utils.data import Dataset
from torch import nn


torch.manual_seed(1)

class BILSTM_CRF(nn.Module):
    def __init__(self,vcab_size,tag2index,emb_dim,hidden_dim,batch_size, use_cuda):
        super(BILSTM_CRF,self).__init__()
        self.vcab_size=vcab_size
        self.tag2index=tag2index
        self.num_tags=len(tag2index)
        self.emb_dim=emb_dim
        self.hidden_dim=hidden_dim
        self.batch_size=batch_size
        self.use_cuda = use_cuda
        self.use_cuda=torch.cuda.is_available()
        self.embed=nn.Embedding(num_embeddings=vcab_size,embedding_dim=emb_dim)  # b,100,128
        # ->100,b,128
        self.bilstm=nn.LSTM(input_size=emb_dim,hidden_size=hidden_dim,num_layers=1,bidirectional=True,dropout=0.1)  # 100,b,256*2
        self.conv1 = nn.Sequential(
            # b,1,100,128
            nn.Conv2d(1, 128, (1, emb_dim),padding=0),  # b,128,100,1
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(1, 128, (3, emb_dim+2), padding=1),  # b,128,100,1
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(1, 128, (5, emb_dim+4), padding=2),  # b,128,100,1
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        #b,128*3,100,1->100,b,128*3
        self.linear1 = nn.Linear(hidden_dim * 2+128*3,hidden_dim)
        self.drop=nn.Dropout(0.2)
        self.classfy=nn.Linear(hidden_dim,self.num_tags)#100*b,10
        #->100,b,10
        # init transitions
        self.start_transitions = nn.Parameter(torch.Tensor(self.num_tags))#i表示出发，j表示到达
        self.end_transitions = nn.Parameter(torch.Tensor(self.num_tags))#i表示到达，j表示出发
        self.transitions = nn.Parameter(torch.Tensor(self.num_tags, self.num_tags))#i表示出发，j表示到达
        nn.init.uniform(self.start_transitions, -0.1, 0.1)
        nn.init.uniform(self.end_transitions, -0.1, 0.1)
        nn.init.uniform(self.transitions, -0.1, 0.1)

    def init_hidden(self,batch_size):#作为初始化传入lstm的隐含变量
        h_h=Variable(torch.randn(2,batch_size,self.hidden_dim))
        h_c=Variable(torch.randn(2,batch_size,self.hidden_dim))
        if self.use_cuda:
            h_h=h_h.cuda()
            h_c=h_c.cuda()
        return (h_h,h_c)

    def get_bilstm_out(self,x):  # 计算bilstm的输出
        batch_size = x.size(0)
        emb = self.embed(x)

        # cnn输出
        emb_cnn=emb.unsqueeze(1)
        cnn1=self.conv1(emb_cnn)
        cnn2=self.conv2(emb_cnn)
        cnn3=self.conv3(emb_cnn)
        cnn_cat=torch.cat((cnn1,cnn2,cnn3),1)
        cnn_out=cnn_cat.squeeze().permute(2,0,1)  # 100,b,128*3

        emb_rnn=emb.permute(1,0,2)
        init_hidden=self.init_hidden(batch_size)
        lstm_out,hidden=self.bilstm(emb_rnn,init_hidden)

        cat_out=torch.cat((cnn_out,lstm_out),2)  # 100,b,128*3+256*2
        s,b,h=cat_out.size()
        cat_out=cat_out.view(s*b,h)
        cat_out=self.linear1(cat_out)
        cat_out=self.drop(cat_out)
        cat_out=self.classfy(cat_out)
        cat_out=cat_out.view(s,b,-1)
        # out=out.permute(1,0,2)
        return cat_out

    def _log_sum_exp(self,tensor,dim):
        # Find the max value along `dim`
        offset, _ = tensor.max(dim)#b,m
        # Make offset broadcastable
        broadcast_offset = offset.unsqueeze(dim)#b,1,m
        # Perform log-sum-exp safely
        safe_log_sum_exp = torch.log(torch.sum(torch.exp(tensor - broadcast_offset), dim))#b,m
        # Add offset back
        return offset + safe_log_sum_exp

    def get_all_score(self,emissions,mask):#计算所有路径的总分#s,b,h
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (batch_size,seq_length)
        seq_length = emissions.size(0)
        mask = mask.permute(1,0).contiguous().float()

        log_prob = self.start_transitions.view(1, -1) + emissions[0]  # b,m,所有从start出发的路径s0

        for i in range(1, seq_length):
            broadcast_log_prob = log_prob.unsqueeze(2)  # b,m,1
            broadcast_transitions = self.transitions.unsqueeze(0)  #1,m,m
            broadcast_emissions = emissions[i].unsqueeze(1)  # b,1,m

            score = broadcast_log_prob + broadcast_transitions \
                    + broadcast_emissions  # b,m,m

            score = self._log_sum_exp(score, 1)  # b,m即为si

            log_prob = score * mask[i].unsqueeze(1) + log_prob * (1. - mask[i]).unsqueeze(
                1)  # mask为0的保持不变，mask为1的更换score

        # End transition score
        log_prob += self.end_transitions.view(1, -1)
        # Sum (log-sum-exp) over all possible tags
        return self._log_sum_exp(log_prob, 1)  # (batch_size,)返回最终score

    def get_real_score(self,emissions,mask,tags):#计算真实路径得分
        # emissions: (seq_length, batch_size, num_tags)
        # tags: (batch_size,seq_length)
        # mask: (batch_size,seq_length)
        seq_length = emissions.size(0)#s
        mask = mask.permute(1,0).contiguous().float()
        tags=tags.permute(1,0).contiguous()

        # Start transition score
        llh = self.start_transitions[tags[0]]  # (batch_size,),T(start->firstTag)

        for i in range(seq_length - 1):
            cur_tag, next_tag = tags[i], tags[i+1]
            # Emission score for current tag
            llh += emissions[i].gather(1, cur_tag.view(-1, 1)).squeeze(1) * mask[i]#(b,1)->b->b*mask，上一轮score+当前发射概率
            # Transition score to next tag
            transition_score = self.transitions[cur_tag.data, next_tag.data]#当前到下一轮的转换概率
            # Only add transition score if the next tag is not masked (mask == 1)
            llh += transition_score * mask[i+1]#若下一轮为padding则不转换

        # Find last tag index
        last_tag_indices = mask.long().sum(0) - 1  # (batch_size,)计算每个序列真实长度
        last_tags = tags.gather(0, last_tag_indices.view(1, -1)).squeeze(0)#b,最后一个非padding的标签id

        # End transition score
        llh += self.end_transitions[last_tags]#加上从最后一个非padding标签到end的转换概率
        # Emission score for the last tag, if mask is valid (mask == 1)
        llh += emissions[-1].gather(1, last_tags.view(-1, 1)).squeeze(1) * mask[-1]#考虑最后一个seq为有效id

        return llh#b

    def neg_log_likelihood(self,feats,tags,mask):
        #feats:  bilstm的输出#100,b,10
        batch_size=feats.size(1)
        all_score=self.get_all_score(feats,mask)#所有路径总分b
        real_score=self.get_real_score(feats,mask,tags)#真实路径得分b
        loss=(all_score.view(batch_size,1)-real_score.view(batch_size,1)).sum()/batch_size
        return loss #目标是最小化这个值，即最大化没log前的真实占总的比例

    def viterbi_decode(self, emissions,mask):
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (batch_size,seq_length)
        seq_length=emissions.size(0)
        batch_size=emissions.size(1)
        num_tags=emissions.size(2)
        length_mask = torch.sum(mask, dim=1).view(batch_size, 1).long()  # 真实序列长度b,1
        mask=mask.permute(1,0).contiguous().float()#s,b

        viterbi_history=[]
        viterbi_score = self.start_transitions.view(1, -1) + emissions[0]  # b,m,所有从start出发的路径s0

        for i in range(1, seq_length):
            broadcast_viterbi_score = viterbi_score.unsqueeze(2)  # b,m,1
            broadcast_transitions = self.transitions.unsqueeze(0)  #1,m,m
            broadcast_emissions = emissions[i].unsqueeze(1)  # b,1,m

            score = broadcast_viterbi_score + broadcast_transitions \
                    + broadcast_emissions  # b,m,m

            best_score,best_path = torch.max(score, 1)  # b,m即为si
            viterbi_history.append(best_path*mask[i].long().unsqueeze(1))#将带0pading的路径加进来
            viterbi_score = best_score * mask[i].unsqueeze(1) + viterbi_score * (1. - mask[i]).unsqueeze(
                1)  # mask为0的保持不变，mask为1的更换score
        viterbi_score+=self.end_transitions.view(1,-1)#b,m
        best_score,last_path=torch.max(viterbi_score,1)#b
        last_path=last_path.view(-1,1)#b,1
        last_position = (length_mask.contiguous().view(batch_size, 1, 1).expand(batch_size, 1, num_tags) - 1).contiguous()  # 最后一个非padding的位置b,1->b,1,m
        pad_zero = Variable(torch.zeros(batch_size, num_tags)).long()
        if self.use_cuda:
            pad_zero = pad_zero.cuda()
        viterbi_history.append(pad_zero)#(s-1,b,m)->(s,b,m)
        viterbi_history = torch.cat(viterbi_history).view(-1, batch_size, num_tags)  # s,b,m
        insert_last = last_path.view(batch_size, 1, 1).expand(batch_size, 1, num_tags) #要将最后的路径插入最后的真实位置b,1,m
        viterbi_history = viterbi_history.transpose(1, 0).contiguous()  # b,s,m
        viterbi_history.scatter_(1, last_position, insert_last)  # 将最后位置的路径统一改为相同路径b,s,m（back_points中的某些值改变了）
        viterbi_history = viterbi_history.transpose(1, 0).contiguous()  # s,b,m
        decode_idx = Variable(torch.LongTensor(seq_length, batch_size))#最后用来记录路径的s,b
        if self.use_cuda:
            decode_idx = decode_idx.cuda()
        # decode_idx[-1] = 0
        for idx in range(len(viterbi_history)-2,-1,-1):
            last_path=torch.gather(viterbi_history[idx],1,last_path)
            decode_idx[idx]=last_path.data.squeeze(1)
        decode_idx=decode_idx.transpose(1,0)#b,s
        return decode_idx

    def forward(self, feats,mask):
        #feats    #bilstm的输出#100.b.10
        best_path=self.viterbi_decode(feats,mask)#最佳路径b,s
        return best_path

def getTest_xy(filepath, word2index, tag2index):
    MAX_LEN = 100
    data=open(filepath,'r')
    test_x = []  # 总x测试集
    test_y=[]  # 总y测试集
    test_word=[]  # 所有句话的词
    sen_x = []  # 每次存一句话的id组
    sen_y=[]    # 每次存一句话的标签id组
    sen_word=[]  # 一句话的词
    end= 1
    # 将数据按每句话分出来
    for line in data:
        line = line.rstrip()
        if (line == "" or line == "\n" or line == "\r\n"):  # 一句话结束了
            if end:
                end = 0
                test_x.append(sen_x)
                sen_x = []
                test_y.append(sen_y)
                sen_y=[]
                test_word.append(sen_word)
                sen_word=[]
            continue
        end = 1
        line = line.split(' ')
        if len(line)==2:
            sen_word.append(line[0])
            if line[0] in word2index:  # 如果在词典中有该词，将id给sen_x
                # print(line)
                sen_x.append(word2index[line[0]])
                sen_y.append(tag2index[line[1]])
            else:  # 如果没有则设为未识别
                sen_x.append(1)
                sen_y.append(tag2index[line[1]])

    # 开始对每句话进行裁剪，主要是最大长度的限制
    test_x_cut = [] #  每个分割的词id
    test_y_cut = []  # 每个分割的标签id
    test_mask = []
    test_x_len = []  # 每句话本身的长度（不填充的长度）
    test_x_cut_word = []  # 所有分割出的词
    count = 0  # 用于样本计数
    test_x_fenge = []  # 用于记分割了的样本序号
    for i in range(len(test_x)):  # i<--第i个句子
        if len(test_x[i]) <= MAX_LEN:  # 如果句子长度小于max_sen_len
            test_x_cut.append(test_x[i])
            test_y_cut.append(test_y[i])
            test_mask.append([1]*len(test_x[i]))
            test_x_len.append(len(test_x[i]))
            test_x_cut_word.append(test_word[i])  # 将这句话的所有字加进去
            count += 1
            continue
        while len(test_x[i]) > MAX_LEN:  # 超过100，使用标点符号拆分句子，将前面部分加入训练集，若后面部分仍超过100，继续拆分
            flag = False
            for j in reversed(range(MAX_LEN)):  # 反向访问，99、98、97...
                if test_x[i][j] == word2index[','] or test_x[i][j] == word2index['、']:
                    test_x_cut.append(test_x[i][:j+1])
                    test_y_cut.append(test_y[i][:j+1])
                    test_mask.append([1]*(j+1))
                    test_x_len.append(j+1)
                    test_x_cut_word.append(test_word[i][:j+1])
                    test_x[i] = test_x[i][j+1:]
                    test_y[i] = test_y[i][j+1:]
                    test_word[i] = test_word[i][j+1:]  # 将词向后滑
                    # test_x_cut_word[i] = test_word[i][j+1:]
                    test_x_fenge.append(count)
                    count += 1
                    break
                if j == 0:  # 拆分不了
                    flag = True
            if flag:
                test_x_cut.append(test_x[i][:MAX_LEN])
                test_y_cut.append(test_y[i][:MAX_LEN])
                test_mask.append([1]*MAX_LEN)
                test_x_len.append(MAX_LEN)
                test_x_cut_word.append(test_word[i][:MAX_LEN])
                test_x[i] = test_x[i][MAX_LEN:]
                test_y[i]=test_y[i][MAX_LEN:]
                test_word[i] = test_word[i][MAX_LEN:]
                # test_x_cut_word[i] = test_word[i][MAX_LEN:]
                test_x_fenge.append(count)
                count+=1
        if len(test_x[i]) <= MAX_LEN:  # 如果句子长度小于max_sen_len，最后没有超过100的直接加入
            test_x_cut.append(test_x[i])
            test_y_cut.append(test_y[i])
            test_mask.append([1]*len(test_x[i]))
            test_x_len.append(len(test_x[i]))
            test_x_cut_word.append(test_word[i])
            count += 1

    # 给每段分割填充0
    for i in range(len(test_x_cut)):
        if len(test_x_cut[i]) < MAX_LEN:
            tlen = len(test_x_cut[i])
            for j in range(MAX_LEN - tlen):
                test_x_cut[i].append(0)

    for i in range(len(test_y_cut)):
        if len(test_y_cut[i]) < MAX_LEN:
            tlen = len(test_y_cut[i])
            for j in range(MAX_LEN - tlen):
                test_y_cut[i].append(0)

    for i in range(len(test_mask)):
        if len(test_mask[i]) < MAX_LEN:
            tlen = len(test_mask[i])
            for j in range(MAX_LEN - tlen):
                test_mask[i].append(0)
    #转化LongTensor
    test_x_cut=torch.LongTensor(test_x_cut)
    test_y_cut=torch.LongTensor(test_y_cut)
    test_mask=torch.ByteTensor(test_mask)
    return test_x_cut,test_y_cut,test_mask,test_x_len,test_x_cut_word,test_x_fenge

def formatData(data):
    fout = open(r'nerApi/data/test.txt', 'w')

    for c in data:
        if c==' ':
            fout.write('\n')
        else:
            fout.write(c+' 0\n')


    fout.write('\n')
    fout.write('# 0\n\n')







    fout.close()

class TextDataSet(Dataset):
    def __init__(self,inputs,outputs,masks):
        self.inputs,self.outputs,self.masks=inputs,outputs,masks
    def __getitem__(self, item):
        return self.inputs[item],self.outputs[item],self.masks[item]
    def __len__(self):
        return len(self.inputs)

def getData(data,topic:str):

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    use_cuda = torch.cuda.is_available()
    formatData(data)
    # word2index, index2word, tag2index, index2tag = data_preprocess.get_dic()
    # print('加载词典')
    word2index = pickle.load(open('nerApi/{}/word2index'.format(topic), 'rb'))
    index2word = pickle.load(open('nerApi/{}/index2word'.format(topic), 'rb'))
    tag2index = pickle.load(open('nerApi/{}/tag2index'.format(topic), 'rb'))
    index2tag = pickle.load(open('nerApi/{}/index2tag'.format(topic), 'rb'))
    # test_x_cut, test_y_cut, test_mask, test_x_len, test_x_cut_word, test_x_fenge = data_preprocess.getTest_xy(
    #     # './data/test.txt')
    test_x_cut, test_y_cut, test_mask, test_x_len, test_x_cut_word, test_x_fenge = \
        getTest_xy(r'nerApi/data/test.txt', word2index, tag2index)
    # print(test_x_cut, test_y_cut, test_mask, test_x_len, test_x_cut_word, test_x_fenge)
    testDataSet = TextDataSet(test_x_cut, test_y_cut, test_mask)

    testDataLoader = DataLoader(testDataSet, batch_size=16, shuffle=False, num_workers=8)
    MAXLEN = 100
    vcab_size = len(word2index)
    emb_dim = 128
    hidden_dim = 256
    num_epoches = 20
    batch_size = 16

    if use_cuda:
        model = BILSTM_CRF(vcab_size, tag2index, emb_dim, hidden_dim, batch_size, use_cuda).cuda()
    else:
        model = BILSTM_CRF(vcab_size, tag2index, emb_dim, hidden_dim, batch_size, use_cuda)

    model.load_state_dict(torch.load('nerApi/{}/best_model.pth'.format(topic)))

    # model.eval()
    test_loss = 0
    test_acc = 0
    batch_len_all = 0
    prepath_all = []  # 所有batch的路径综合

    for i, data in enumerate(testDataLoader):
        # print(i)
        x, y, mask = data
        # print(x)
        # print(y)
        # print(mask)
        batch_len = len(x)
        batch_len_all += batch_len
        if use_cuda:
            x = Variable(x, volatile=True).cuda()
            y = Variable(y, volatile=True).cuda()
            mask = Variable(mask, volatile=True).cuda()
        else:
            x = Variable(x, volatile=True)
            y = Variable(y, volatile=True)
            mask = Variable(mask, volatile=True)
        feats = model.get_bilstm_out(x)
        loss = model.neg_log_likelihood(feats, y, mask)
        test_loss += loss.data[0]
        prepath = model(feats, mask)  # b,s
        prepath_all.append(prepath)
        # pre_y = prepath.masked_select(mask)
        # true_y = y.masked_select(mask)
        # acc_num = (pre_y == true_y).data.sum()
        # acc_pro = float(acc_num) / len(pre_y)
        # test_acc += acc_pro

    # print('test loss is:{:.6f},test acc is:{:.6f}'.format(test_loss / (len(testDataLoader)), test_acc / (len(testDataLoader))))

    # 写入结果文件
    prepath_all = torch.cat(prepath_all).data
    # data_preprocess.write_result_to_file('./data/result.txt', prepath_all, test_x_len, test_x_cut_word, test_x_fenge)

    # y[i][j] : i句j标签
    result = []
    for i1 in range(prepath_all.shape[0]):  # 样本数
        # print('------------------', len(test_x_cut_word[i1]))
        flag, s = True, ''
        curTag = []

        for i2 in range(test_x_len[i1]):  # 每个样本的真实长度
            tag_id = prepath_all[i1][i2]
            word = test_x_cut_word[i1][i2]

            if word=='#': word = ''
            if tag_id in index2tag:
                tag = index2tag[tag_id]
            else:
                tag = '0'
            result.append(tag)
            # CHY
            if flag and tag[0]=='B':
                curTag = tag.split('-')[1:]
                s += '<span class='+''.join(curTag)+'>' + word
                flag = False
            elif (not flag and tag[0] == 'O') or (flag and tag[0] == 'B'):

                s += '[' + ''.join(curTag) + ']</span>'+word
                flag = True
                curTag = []
            elif not flag and i2==test_x_len[i1]-1:
                s += word + '[' + ''.join(curTag) + ']</span>'

            else:
                s += word

    return result

