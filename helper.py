from itertools import takewhile
from collections import defaultdict

label_dict = {'F1':'股票','F2':'股票代码','F3':'理财产品','C1':'品牌','C2':'车系','C3':'车型','C4':'零部件'}

def split(line):
    result = []
    temp_result = []
    while line:
        # print(line)
        head,*tail = line
        temp_result.append(head)
        other = list(takewhile(lambda x: x[1][0] != 'B',tail))
        # print(other)
        temp_result.extend(other)
        result.append(temp_result)
        temp_result = []
        line = tail[len(other):]
    return result


def convert(value):
    return (''.join(list(map(lambda x: x[0],value))),value[0][1][2:])

def helper(a,b):
    result = defaultdict(list)
    for x,y in list(map(convert,split(list(filter(lambda x: (x[1]!='0' and x[1]!='pad'),zip(a,b)))))):
        result[y].append(x)
    return [{'label':label_dict[k],'value':'    '.join(v)} for k,v in result.items()]

# res = helper('烟锁池塘柳',['0','B-F2','I-F2','0','B-F1'])
# print(res)