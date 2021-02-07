import json
import collections

def getlabeldict(path):
    decoder = json.JSONDecoder(object_pairs_hook=collections.OrderedDict)
    with open(path) as file:
        labeldict = decoder.decode(file.read())
    return labeldict

def jsonfileparser(filepath, encoding='utf-8'):
    print('parse information from json file')
    with open(filepath, encoding=encoding) as file:
        dict = json.load(file, object_pairs_hook=collections.OrderedDict)

    return dict