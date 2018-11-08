from itertools import takewhile
from collections import defaultdict

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
    for x,y in list(map(convert,split(list(filter(lambda x: x[1]!='0',zip(a,b)))))):
        result[y].append(x)
    return result

# res = helper('烟锁池塘柳',['0','B-F2','I-F2','0','B-F1'])
# print(res)