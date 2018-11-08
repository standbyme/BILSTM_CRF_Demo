from itertools import takewhile
import time

line = [('财','B-F3'),('富','I-F3'),('宝','B-F3')]

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
    return (''.join(list(map(lambda x: x[0],value))),value[0][1])

print(list(map(convert,split(line))))