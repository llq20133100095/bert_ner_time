#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/30 13:52
# @Author  : honeyding
# @File    : check_corpus.py
# @Software: PyCharm
import pandas as pd
import codecs

import random
import sys
reload(sys)
sys.setdefaultencoding('utf8')
def str_type(type, flag, str):
    if type == u'time_point':
        str = str + flag + '-POINT\t'
    elif type == u'time_period':
        str = str + flag + '-PERIOD\t'
    elif type == u'time_duration':
        str = str + flag + '-DURATION\t'
    elif type == u'time_fuzzy':
        str = str + flag + '-FUZZY\t'
    elif type == u'time_prope':
        str = str + flag + '-PROPE\t'
    elif type == u'time_prper':
        str = str + flag + '-PROPE\t'
    return str

def str_dim(abs_rel, is_refer, is_freq, str):
    if abs_rel == 'relative':
        str = str + 'REL\t'
    elif abs_rel == 'absolute':
        str = str + 'ABS\t'
    else:
        str = str + 'O\t'

    if is_refer:
        str = str + 'T\t'
    else:
        str = str + 'O\t'

    if is_freq:
        str = str + 'T\n'
    else:
        str = str + 'O\n'
    return str

def setdict(dict, sentence, entity, f_error, is_refer, abs_rel, is_freq, type):
    e = 0
    for val in dict.values():
        if val['end'] > e:
            e = val['end']

    start = sentence.find(entity,e)
    if start < 0:
        f_error.write(u'sentence:{} entity:{} 顺序不对\n'.format(sentence, entity))
    end = start + len(entity)
    if end > len(sentence):
        return dict
    key = start
    if key in dict.keys():
        start = sentence.find(entity, end+1)
        key = start
        if start != -1:
            end = start + len(entity)
            if end > len(sentence):
                return dict
            if key in dict.keys():
                start = sentence.find(entity, end+1)
                key = start
                if start != -1:
                    end = start + len(entity)
                    if end > len(sentence):
                        return dict
                    if key in dict.keys():
                        f_error.write(u'sentence:{} entity:{} 重复实体不止两次\n'.format(sentence, entity))
                    else:
                        dict_val = {u'end': end, u"entity": entity, u"type": type, u"abs_rel": abs_rel, u"is_refer": is_refer,
                    u"is_freq": is_freq}
                        dict[key] = dict_val
                else:
                    f_error.write(u'sentence:{} entity:{} 实体位置不对\n'.format(sentence, entity))
            else:
                dict_val = {u'end': end, u"entity": entity, u"type": type, u"abs_rel": abs_rel, u"is_refer": is_refer,
                    u"is_freq": is_freq}
                dict[key] = dict_val
        else:
            f_error.write(u'sentence:{} entity:{} 实体位置不对\n'.format(sentence, entity))

    else:
        dict_val = {u'end': end, u"entity": entity, u"type": type, u"abs_rel": abs_rel, u"is_refer": is_refer,
                    u"is_freq": is_freq}
        dict[key] = dict_val

    return dict

def print_data(dict, i, f_train, sen):
    list_index = dict.keys()
    list_index = sorted(list_index)
    end_last = 0
    for start in list_index:
        value = dict[start]
        end = value['end']
        _entity = value['entity'].strip()
        _type = value['type']
        _abs_rel = value['abs_rel']
        _is_refer = value['is_refer']
        _is_freq = value['is_freq']
        if end_last != start and end_last < start:
            for k in range(end_last, start):
                f_train.write(u'{}\t{}\t{}\t{}\t{}\n'.format(sen[k], u'O', u'O', u'O', u'O'))
        if end - start == 1:
            str = sen[start] + '\t'
            str = str_type(_type, 'S', str)
            str = str_dim(_abs_rel, _is_refer, _is_freq, str)

            f_train.write(str)
            end_last = end
        else:
            for k in range(len(_entity)):
                str = sen[start + k] + '\t'
                if k == 0:
                    str = str_type(_type, 'B', str)

                elif k == len(_entity) - 1:
                    str = str_type(_type, 'E', str)
                else:
                    str = str_type(_type, 'I', str)

                str = str_dim(_abs_rel, _is_refer, _is_freq, str)

                f_train.write(str)
                end_last = start + k + 1

    if end_last < len(sen):
        for k in range(end_last, len(sen)):
            f_train.write(u'{}\t{}\t{}\t{}\t{}\n'.format(sen[k], u'O', u'O', u'O', u'O'))

    f_train.write('\n')

def generate_dataset(file,filetxt):

    f_train = codecs.open(filetxt, u'w', 'utf-8')
    f_error = codecs.open(u'./glue/resources/error.txt', u'w', 'utf-8')

    ncount = 0
    f = pd.read_csv(file, encoding=u'utf-8')
    sen = ''
    dict = {}
    for i, (idx, df) in enumerate(f.iterrows()):
        ncount = ncount +1
        sentence = df[u"sentence"]
        entity = df[u"entity"]
        type = df[u"type"]
        abs_rel = df[u"abs_rel"]
        is_refer = df[u"is_refer"]
        is_freq = df[u"is_freq"]

        if pd.isnull(sentence) or sen == '':
            if pd.isnull(sentence):
                sentence = sen
            else:
                sen = sentence
            if pd.isnull(entity):
                continue
            else:
                entity = entity.strip()
                try:
                    if entity not in sentence:
                        continue
                    else:
                        dict = setdict(dict, sentence, entity, f_error, is_refer, abs_rel, is_freq, type)
                except:
                    continue

        elif sen != '':
            print_data(dict, ncount, f_train, sen)
            sen = sentence
            dict = {}

            if pd.isnull(entity):
                continue
            else:
                try:
                    entity = entity.strip()
                    if entity not in sentence:
                        continue
                    else:
                        dict = setdict(dict, sentence, entity, f_error, is_refer, abs_rel, is_freq, type)
                except:
                    continue
        if i == len(f[u"sentence"]) -1:
            print_data(dict, ncount, f_train, sen)

    f_train.close()
    f_error.close()


def check_corpus():
    file_list = [u'../glue/resources/time_corpus1-5500.csv']
    f_error = codecs.open(u'../glue/resources/error_entities.txt', u'w', 'utf-8')
    indexs, sentences, entities = [],[],[]
    for file in file_list:
        f = pd.read_csv(file, encoding=u'utf-8')
        sen = ''
        _index = None
        for i, (idx, df) in enumerate(f.iterrows()):
            index = df[u"index"]
            sentence = df[u"sentence"]
            entity = df[u"entity"]

            if pd.isnull(sentence):
                sentence = sen
            else:
                sen = sentence
            if pd.isnull(index):
                index = _index
            else:
                _index = index
                print(index)

            if pd.isnull(entity):
                continue
            else:
                try:
                    entity = entity.strip()
                    if entity not in sentence:
                        indexs.append(int(index))
                        sentences.append(sentence)
                        entities.append(entity)
                        f_error.write(u'index: {}  entity: {}  sentence: {}  \n'.format(int(index), entity, sentence))
                except:
                    indexs.append(int(index))
                    sentences.append(sentence)
                    entities.append(entity)
                    f_error.write(u'index: {}  entity: {}   sentence: {}\n'.format(int(index), entity, sentence))

def setdict_sentence(dict, index, sentence, entity, f_error, is_refer, abs_rel, is_freq, type):
    if sentence in dict.keys():
        dict_val = {u'index': index, u"entity": entity, u"type": type, u"abs_rel": abs_rel, u"is_refer": is_refer,
                u"is_freq": is_freq}
        val = dict[sentence]
        val.append(dict_val)
        dict[sentence] = val

    else:
        dict_val = {u'index': index, u"entity": entity, u"type": type, u"abs_rel": abs_rel, u"is_refer": is_refer,
                    u"is_freq": is_freq}
        val = []
        val.append(dict_val)
        dict[sentence] = val

    return dict

def split_train_test():
    file_list = [u'./resources/time_corpus.csv']
    f_error = codecs.open(u'./resources/error.txt', u'w', 'utf-8')

    ncount = 0
    for file in file_list:
        f = pd.read_csv(file, encoding=u'utf-8')
        sen = ''
        dict = {}
        for i, (idx, df) in enumerate(f.iterrows()):
            ncount = ncount + 1
            index = df[u"index"]
            sentence = df[u"sentence"]
            entity = df[u"entity"]
            type = df[u"type"]
            abs_rel = df[u"abs_rel"]
            is_refer = df[u"is_refer"]
            is_freq = df[u"is_freq"]

            if pd.isnull(sentence):
                if pd.isnull(sentence):
                    sentence = sen
                else:
                    sen = sentence
                if pd.isnull(entity):
                    print(sentence)
                else:
                    entity = entity.strip()
                    try:
                        if entity not in sentence:
                            print(sentence)
                        else:
                            dict = setdict_sentence(dict, index, sentence, entity, f_error, is_refer, abs_rel, is_freq, type)
                    except:
                        print(sentence)

            else:
                sen = sentence
                if pd.isnull(entity):
                    dict[sentence] = [{'index':index,'entity':'','type':'','abs_rel':'','is_refer':'','is_freq':''}]
                else:
                    try:
                        entity = entity.strip()
                        if entity not in sentence:
                            print(sentence)
                        else:
                            dict = setdict_sentence(dict, index, sentence, entity, f_error, is_refer, abs_rel, is_freq, type)
                    except:
                        print(sentence)

    sentences = dict.keys()
    random.shuffle(sentences)
    random.shuffle(sentences)
    random.shuffle(sentences)

    import csv
    # 头数据
    fileHeader = ['index','sentence','entity','type','abs_rel','is_refer','is_freq']
    # 写入数据
    csvFile_train = open("./resources/train.csv", "w")
    writer_train = csv.writer(csvFile_train)
    # 分批写入
    writer_train.writerow(fileHeader)

    # 写入数据
    csvFile_train5000 = open("./resources/train5000.csv", "w")
    writer_train5000 = csv.writer(csvFile_train5000)
    # 分批写入
    writer_train5000.writerow(fileHeader)

    # 写入数据
    csvFile_train10000 = open("./resources/train10000.csv", "w")
    writer_train10000 = csv.writer(csvFile_train10000)
    # 分批写入
    writer_train10000.writerow(fileHeader)

    # 写入数据
    csvFile_validate = open("./resources/validate.csv", "w")
    writer_validate = csv.writer(csvFile_validate)
    # 分批写入
    writer_validate.writerow(fileHeader)

    # 写入数据
    csvFile_test = open("./resources/test.csv", "w")
    writer_test = csv.writer(csvFile_test)
    # 分批写入
    writer_test.writerow(fileHeader)
    count = 5000
    for i, sen in enumerate(sentences):
        if i%10 == 2:
            list_val_validate = dict[sen]
            if len(list_val_validate) == 0:
                list = ['', sen, '', '', '', '', '']
                writer_validate.writerow(list)
            else:
                for j, val in enumerate(list_val_validate):
                   index = val['index']
                   entity = val['entity']
                   type = val['type']
                   abs_rel = val['abs_rel']
                   is_freq = val['is_freq']
                   is_refer = val['is_refer']
                   if j == 0:
                       list = [str(index),sen,entity,type,abs_rel,is_refer,is_freq]
                       writer_validate.writerow(list)
                   else:
                       list = ['', '', entity, type, abs_rel, is_refer, is_freq]
                       writer_validate.writerow(list)

        elif i % 10 == 8:
            list_val_validate = dict[sen]
            if len(list_val_validate) == 0:
                list = ['', sen, '', '', '', '', '']
                writer_test.writerow(list)
            else:
                for j, val in enumerate(list_val_validate):
                    index = val['index']
                    entity = val['entity']
                    type = val['type']
                    abs_rel = val['abs_rel']
                    is_freq = val['is_freq']
                    is_refer = val['is_refer']
                    if j == 0:
                        list = [str(index), sen, entity, type, abs_rel, is_refer, is_freq]
                        writer_test.writerow(list)
                    else:
                        list = ['', '', entity, type, abs_rel, is_refer, is_freq]
                        writer_test.writerow(list)
        else:
            if i % 10 == 3 or i % 10 == 4 or i % 10 == 5:
                count = count - 1
                list_val_validate = dict[sen]
                if len(list_val_validate) == 0:
                    list = ['', sen, '', '', '', '', '']
                    if count >= 0:
                        writer_train5000.writerow(list)
                    writer_train10000.writerow(list)
                else:
                    for j, val in enumerate(list_val_validate):
                        index = val['index']
                        entity = val['entity']
                        type = val['type']
                        abs_rel = val['abs_rel']
                        is_freq = val['is_freq']
                        is_refer = val['is_refer']
                        if j == 0:
                            list = [str(index), sen, entity, type, abs_rel, is_refer, is_freq]
                            if count >= 0:
                                writer_train5000.writerow(list)
                            writer_train10000.writerow(list)
                        else:
                            list = ['', '', entity, type, abs_rel, is_refer, is_freq]
                            if count >= 0:
                                writer_train5000.writerow(list)
                            writer_train10000.writerow(list)

            if i % 10 == 7 or i % 10 == 9:
                list_val_validate = dict[sen]
                if len(list_val_validate) == 0:
                    list = ['', sen, '', '', '', '', '']
                    writer_train10000.writerow(list)
                else:
                    for j, val in enumerate(list_val_validate):
                        index = val['index']
                        entity = val['entity']
                        type = val['type']
                        abs_rel = val['abs_rel']
                        is_freq = val['is_freq']
                        is_refer = val['is_refer']
                        if j == 0:
                            list = [str(index), sen, entity, type, abs_rel, is_refer, is_freq]
                            writer_train10000.writerow(list)
                        else:
                            list = ['', '', entity, type, abs_rel, is_refer, is_freq]
                            writer_train10000.writerow(list)

            list_val_validate = dict[sen]
            if len(list_val_validate) == 0:
                list = ['', sen, '', '', '', '', '']
                writer_train.writerow(list)
            else:
                for j, val in enumerate(list_val_validate):
                    index = val['index']
                    entity = val['entity']
                    type = val['type']
                    abs_rel = val['abs_rel']
                    is_freq = val['is_freq']
                    is_refer = val['is_refer']
                    if j == 0:
                        list = [str(index), sen, entity, type, abs_rel, is_refer, is_freq]
                        writer_train.writerow(list)
                    else:
                        list = ['', '', entity, type, abs_rel, is_refer, is_freq]
                        writer_train.writerow(list)

    csvFile_train.close()
    csvFile_validate.close()
    csvFile_test.close()
    f_error.close()

def demo():
    f_demo = codecs.open(u'./resources/output_txt.txt', u'w', 'utf-8')
    count = 20000
    with open(u'./resources/output.txt') as tf:
        lines = tf.readlines()
        for line in lines:
            line_list = line.split('\r')
            for line in line_list:
                line = line.decode('utf-8').strip()
                for i in range(len(line)):
                    if count < 0:
                        break
                    sen = line[i]
                    f_demo.write(sen)
                    f_demo.write(u'\n')
                f_demo.write(u'\n')
                count = count-1




if __name__ == '__main__':
    # writer = pd.ExcelWriter('output.xlsx')
    # df1 = pd.DataFrame(data={'col1': [1, 1], 'col2': [2, 2]})
    # df1.to_excel(writer, 'Sheet1')
    # writer.save()

    # check_corpus()
    # split_train_test()
    # generate_dataset(u'./resources/train.csv',u'./resources/train.txt')
    # generate_dataset(u'../glue/resources/test.csv', u'../glue/resources/test.txt')
    # generate_dataset(u'../glue/resources/validate.csv', u'../glue/resources/validate.txt')
    # generate_dataset(u'./resources/train5000.csv', u'./resources/train5000.txt')
    # generate_dataset(u'./resources/train10000.csv', u'./resources/train10000.txt')
    # generate_dataset(u'./resources/time_corpus.csv', u'./resources/time_corpus.txt')
    # generate_dataset(u'./glue/resources/time_corpus1-5500的副本.csv', u'./glue/resources/time_corpus5501-10000.txt')
    generate_dataset(u'./glue/resources/time_corpus15001-20000的副本.csv', u'./glue/resources/time_corpus15001-20000的副本.txt')


    # demo()