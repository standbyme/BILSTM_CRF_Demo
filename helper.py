from itertools import takewhile
from collections import defaultdict

label_dict = {'M1': '机构名—医院', 'M2': '疾病', 'M3': '科室', 'M4': '药品', 'M5': '症状', 'P1': '人名—明星', 'P2': '体育项目', 'P3': '赛事', 'H1': '小区', 'H2': '楼盘', 'H3': '机构名—开发商', 'H4': '机构名—物业公司', 'H5': '家居', 'H6': '家具品牌', 'D1': '品牌', 'D2': '菜名', 'D3': '食品名', 'D4': '机构名—餐馆名', 'D5': '机构名—商场/购物中心', 'J1': '景点', 'J2': '机构名—酒店', 'J3': '机构名—航空公司',
              'J4': '航班', 'J5': '车次', 'J6': '车站', 'E1': '人名—明星', 'E2': '影视剧', 'E3': '歌曲', 'E4': '专辑', 'E5': '电视节目', 'E6': '网络节目', 'G1': '游戏名', 'G2': '人名—虚拟角色', 'G3': '道具', 'G4': '技能', 'T1': '学校', 'T2': '课程名', 'T3': '书籍名', 'T4': '人名—作者', 'T5': '培训机构', 'F1': '股票', 'F2': '股票代码', 'F3': '理财产品', 'C1': '品牌', 'C2': '车系', 'C3': '车型', 'C4': '零部件'}


def split(line):
    result = []
    temp_result = []
    while line:
        # print(line)
        head, *tail = line
        temp_result.append(head)
        other = list(takewhile(lambda x: x[1][0] != 'B', tail))
        # print(other)
        temp_result.extend(other)
        result.append(temp_result)
        temp_result = []
        line = tail[len(other):]
    return result


def convert(value):
    return (''.join(list(map(lambda x: x[0], value))), value[0][1][2:])


def helper(a, b):
    result = defaultdict(list)
    for x, y in list(map(convert, split(list(filter(lambda x: (x[1] != '0' and x[1] != 'pad'), zip(a, b)))))):
        result[y].append(x)
    return [{'label': label_dict[k], 'value':'    '.join(v)} for k, v in result.items()]

# res = helper('烟锁池塘柳',['0','B-F2','I-F2','0','B-F1'])
# print(res)
