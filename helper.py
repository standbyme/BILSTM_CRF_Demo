from itertools import takewhile
import time

# [('财','B-F3'),('富','I-F3'),('宝','B-F3')]

def split(line):
    result = []
    temp_result = []
    while line:
        # print(line)
        head,*tail = line
        time.sleep(2)
        temp_result.append(head)
        other = list(takewhile(lambda x: x[1][0] != 'B',tail))
        # print(other)
        temp_result.extend(other)
        result.append(temp_result)
        temp_result = []
        line = tail[len(other):]


print(result)