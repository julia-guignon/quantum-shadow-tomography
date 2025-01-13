


def replace_in_list(lst, a, b):
    '''
        Replaces a by b in a list lst. Returns a new list.
    '''
    return list(map(lambda x: x if x!=a else b, lst))

# def flatten_list(lst):
#     '''
#         Converts list of list into a list by flattening it.
#     '''
#     return sum(lst, [])

def get_values_dict_of_dict(dct):
    '''
        Returns a single list of all the values in a dict of dict.
    '''
    return sum([[v for v in el.values()] for el in list(dct.values())], [])

