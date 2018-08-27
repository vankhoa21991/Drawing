import os
import json
import io

def create_encode_decode_file(lbls):
    filelist = {}
    unique_lbls = list(set(lbls))
    for i in range(len(unique_lbls)):
        filelist[unique_lbls[i]]= i


    with io.open('/mnt/DATA/lupin/Drawing/keras_model/data/encode_kanji.json', 'w', encoding='utf8') as json_file:
        json.dump(filelist, json_file, ensure_ascii=False)
