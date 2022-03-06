from functools import reduce
import operator


def delete_keys_from_dict(dictionary, key_list):
    """
    delets set of keys from dictionary
    """
    dictionary = reduce(dict.get, key_list[0:-1], dictionary)
    del dictionary[key_list[-1]]


def nested_dict_pairs_iterator(dict_obj):
    '''
    This function accepts a nested dictionary as argument
    and iterate over all values of nested dictionaries
    '''
    # Iterate over all key-value pairs of dict argument
    for key, value in dict_obj.items():
        # Check if value is of dict type
        if isinstance(value, dict):
            # If value is dict then iterate over all its values
            for pair in nested_dict_pairs_iterator(value):
                yield (key, *pair)
        else:
            # If value is not dict type then yield the value
            yield (key, value)


def setInDict(dataDict, mapList, value):
    """
    sets value to nested dict
    dataDict: dictionary
    mapList: list of keys to access value
    value: any object that can be stored in a dict
    """
    getFromDict(dataDict, mapList[:-1])[mapList[-1]] = value


def getFromDict(data_dict, map_list):
    """
    returns value to nested dict
    dataDict: dictionary
    mapList: list of keys to access value
    """
    return reduce(operator.getitem, map_list, data_dict)


def merge(a, b, path=None):
    "merges b into a"
    if path is None: path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge(a[key], b[key], path + [str(key)])
            elif a[key] == b[key]:
                pass  # same leaf value
            else:
                raise Exception('Conflict at %s' % '.'.join(path + [str(key)]))
        else:
            a[key] = b[key]
    return a


def appendInDictList(data_dict, map_list, value):
    """
    appends value to nested dict with list-values
    """
    getFromDict(data_dict, map_list[:-1])[map_list[-1]].append(value)


def nested_val_set(dic, keys, value):
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    dic[keys[-1]] = value


def compare_dictionaries(dict_1, dict_2, path=""):
    """Compare two dictionaries recursively to find non mathcing elements
    https://stackoverflow.com/questions/27265939/comparing-python-dictionaries-and-nested-dictionaries
    Args:
        dict_1: dictionary 1
        dict_2: dictionary 2

    Returns:

    """

    key_err = 0
    value_err = 0
    old_path = path
    for k in dict_1.keys():
        path = old_path + "[%s]" % k
        if not k in dict_2.keys():
            key_err += 1
        else:
            if isinstance(dict_1[k], dict) and isinstance(dict_2[k], dict):
                ke, ve = compare_dictionaries(dict_1[k], dict_2[k], path)
                key_err += ke
                value_err += ve
            else:
                if dict_1[k] != dict_2[k]:
                    value_err += 1

    for k in dict_2.keys():
        path = old_path + "[%s]" % k
        if not k in dict_1.keys():
            key_err += 1
    return key_err, value_err
