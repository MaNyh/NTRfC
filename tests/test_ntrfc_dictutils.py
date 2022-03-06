from ntrfc.utils.dictionaries.dict_utils import nested_dict_pairs_iterator, delete_keys_from_dict, compare_dictionaries, \
    setInDict, getFromDict, merge


def test_nested_dict_pairs_iterator():
    """
    tests if nested_dict_pairs_iterator returns the right list for a nested dict
    nested_dict_pairs_iterator is used for returning a nested dict as a list
    which comes in handy when handling nested directories with files
    """
    test_dictionary = {"0": {"0": 0, "1": 1}, "1": 1}
    check = [('0', '0', 0), ('0', '1', 1), ('1', 1)]
    assert check == list(nested_dict_pairs_iterator(test_dictionary)), "error"


def test_delete_keys_from_dict():
    test_dictionary = {"a": 0, "b": 1}
    delete_keys_from_dict(test_dictionary, "a")
    assert "a" not in test_dictionary.keys()
    assert "b" in test_dictionary.keys()
    assert test_dictionary["b"] == 1


def test_compare_dictionaries():
    d1 = {"name": {"param_1": 0, "param_2": 3}}
    d2 = {"name": {"param_1": 4, "param_2": 1}}
    ke, ve = compare_dictionaries(d1, d2)
    assert ke == 0
    assert ve == 2


def test_setInDict():
    dict = {"toplevel": {"value_1": 0, "value_2": 2}}
    setInDict(dict, ["toplevel", "value_2"], 3)
    assert dict["toplevel"]["value_2"] == 3


def test_getFromDict():
    """
    sets value to nested dict
    dataDict: dictionary
    mapList: list of keys to access value
    """
    dict = {"toplevel": {"midlevel": {"value_1": 1, "value_2": 0}}}

    assert getFromDict(dict, ["toplevel", "midlevel", "value_2"]) == 0


def test_merge():
    dict_1 = {"toplevel_1": {"value_11": 0}}
    dict_2 = {"toplevel_1": {"value_12": 1}, "toplevel_2": {"value_21": 3}}
    assert merge(dict_1, dict_2) == {'toplevel_1': {'value_11': 0, 'value_12': 1}, 'toplevel_2': {'value_21': 3}}
