from functools import reduce
import operator



def nested_dict_pairs_iterator(dict_obj):
    ''' This function accepts a nested dictionary as argument
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
    """
    #todo: test-method
    getFromDict(dataDict, mapList[:-1])[mapList[-1]] = value

def getFromDict(dataDict, mapList):
    #todo: test-method
    return reduce(operator.getitem, mapList, dataDict)
