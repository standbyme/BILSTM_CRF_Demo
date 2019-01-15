from functools import reduce


def buffer_to_result(buffer, result):
    result.append({"type": buffer["type"], "value": ''.join(buffer["value"])})


def reduce_function(acc, current):
    print(acc)
    print(current)
    value: str = current[0]
    flag = current[1][0]
    type_code: str = None

    if flag == '0':
        type_code = '0'
    else:
        type_code = current[1][2:]

    if acc["in_blank"]:
        if flag == '0':
            acc["buffer"]["value"].append(value)
            acc["buffer"]["type"] = type_code
        elif flag == 'B':
            if acc["buffer"]["value"]:
                buffer_to_result(acc["buffer"], acc["result"])
            acc["buffer"] = {"value": [value], "type": type_code}
            acc["in_blank"] = False
    else:
        if flag == '0':
            buffer_to_result(acc["buffer"], acc["result"])
            acc["in_blank"] = True
            acc["buffer"] = {"value": [value], "type": type_code}
        elif flag == 'B':
            buffer_to_result(acc["buffer"], acc["result"])
            acc["buffer"] = {"value": [value], "type": type_code}
        elif flag == 'I':
            acc["buffer"]["value"].append(value)

    return acc


def helper(a, b):
    init = {
        "buffer": {
            "value": [],
            "type": ""
        },
        "result": [],
        "in_blank": True
    }

    acc = reduce(reduce_function, list(
        filter(lambda x: (x[1] != 'pad'), zip(a, b))), init)
    buffer_to_result(acc["buffer"], acc["result"])

    result = acc["result"]

    return result


# res = helper('烟锁池塘柳', ['0', 'B-F2', 'I-F2', 'B-F2', 'B-F1'])
# print(res)
