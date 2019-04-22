#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/01 22:41
# @Author  : honeyding
# @File    : month.py
# @Software: PyCharm

from bert_ner.run_ner_multicrf import tag_recognize
from common.data import get_eval_demo
from normalization.time_parser_hander import TimeParserHander

class named_entities_hander:

    def tag(self,sentence):
        if not sentence:return
        times = []
        results, text_list, label_list_type, label_list_abs_rel, \
        label_list_ref, label_list_fre = tag_recognize(sentence)
        entities = get_eval_demo(results, text_list, label_list_type, label_list_abs_rel, label_list_ref, label_list_fre)
        timeParserHander = TimeParserHander()
        for entity in entities:
            time = timeParserHander.doParse(entity)
            tup = (entity, time)
            times.append(tup)
        return times



if __name__ == '__main__':
    handle = named_entities_hander()
    sentence = '初九'
    times = handle.tag(sentence)
    for (entity, time) in times:
        print("{")
        print("input:" + sentence)
        print("reuslt:{[")
        print("\tentity:" + entity.entity)
        print("\tstart:" + str(entity.start))
        print("\tend:" + str(entity.end))
        print("\ttype:" + str(entity.type))
        print("\tlunar:" + "True")
        print("\tabs_rel:" + "relative")
        print("\tabs_rel:" + str(entity.abs_rel))
        print("\tis_refer:" + str(entity.is_refer))
        print("\tis_freq:" + str(entity.is_freq))
        if time:
            print("\ttimepoint:" + str(time.date) + '\t')
            print('\tstart:' + str(time.startDate) + '\t')
            print('\tend:' + str(time.endDate) + '\t')
            print("\t]}")
            print("}")