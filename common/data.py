#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/30 13:53
# @Author  : honeyding
# @File    : data.py
# @Software: PyCharm


# -*- coding: utf-8 -*-
from common import entity
import itertools

def get_eval_demo(results, text_list,label_list_type, label_list_abs_rel, label_list_ref, label_list_fre):
    label_map_type = {}
    for (i, label) in enumerate(label_list_type):
        label_map_type[i + 1] = label

    label_map_abs_rel = {}
    for (i, label) in enumerate(label_list_abs_rel):
        label_map_abs_rel[i + 1] = label

    label_map_ref = {}
    for (i, label) in enumerate(label_list_ref):
        label_map_ref[i + 1] = label

    label_map_fre = {}
    for (i, label) in enumerate(label_list_fre):
        label_map_fre[i + 1] = label

    predictions = list(itertools.islice(results, len(text_list)))

    pred_labels_type = []
    pred_labels_abs_rel = []
    pred_labels_ref = []
    pred_labels_fre = []
    for i in range(len(predictions)):
        pred_type = predictions[i]['values_type'][1:len(text_list[i]) + 1]
        pred_abs_rel = predictions[i]['values_abs_rel'][1:len(text_list[i]) + 1]
        pred_ref = predictions[i]['values_ref'][1:len(text_list[i]) + 1]
        pred_fre = predictions[i]['values_fre'][1:len(text_list[i]) + 1]

        pred_ = []
        for l in pred_type:
            if l == 0:
                pred_.append(label_map_type[7])
            else:
                pred_.append(label_map_type[l])
        pred_labels_type.append(pred_)

        pred_ = []
        for l in pred_abs_rel:
            if l == 0:
                pred_.append(label_map_abs_rel[3])
            else:
                pred_.append(label_map_abs_rel[l])
        pred_labels_abs_rel.append(pred_)

        pred_ = []
        for l in pred_ref:
            if l == 0:
                pred_.append(label_map_ref[2])
            else:
                pred_.append(label_map_ref[l])
        pred_labels_ref.append(pred_)

        pred_ = []
        for l in pred_fre:
            if l == 0:
                pred_.append(label_map_fre[2])
            else:
                pred_.append(label_map_fre[l])
        pred_labels_fre.append(pred_)

    pred_entities = get_entity(pred_labels_type, pred_labels_abs_rel, pred_labels_ref, pred_labels_fre, text_list)
    return pred_entities

def get_entity(type_tag, relab_tag, ref_tag, fre_tag, char_seq):
    entities = get_type_entity(char_seq, type_tag)
    entities = get_Dimension_info(entities, fre_tag, ref_tag, relab_tag)
    return entities

def get_Dimension_info(entities, fre_tags, ref_tags, relab_tags):
    for entity, fre_tag_list, ref_tag_list, relab_tag_list in zip(entities, fre_tags, ref_tags, relab_tags):
            start = entity.start
            end = entity.end
            fre = list(filter(lambda x: True if fre_tag_list[x] == u'T' else False, range(start,end)))
            if len(fre) == end - start:
                entity.is_freq = True
            ref = list(filter(lambda x: True if ref_tag_list[x] == u'T' else False, range(start, end)))
            if len(ref) == end - start:
                entity.is_refer = True
            abs = list(filter(lambda x: True if relab_tag_list[x] == u'ABS' else False, range(start, end)))
            if len(abs) == end - start:
                entity.abs_rel = 'absolute'
            rel = list(filter(lambda x: True if relab_tag_list[x] == u'REL' else False, range(start, end)))
            if len(rel) == end - start:
                entity.abs_rel = 'relative'
    return entities


type_dict = {'POINT':'time_point', 'PERIOD':'time_period', 'DURATION':'time_duration', 'FUZZY':'time_fuzzy', 'PROPE':'time_prope'}
def get_type_entity(char_seqs, type_tags):
    list_entities = []
    for i, (type_tag, char_seq) in enumerate(zip(type_tags, char_seqs)):
        entity_seq = []
        instance = entity.entity()
        for i, (char, tag) in enumerate(zip(char_seq, type_tag)):
            if "B-" in tag:
                instance = entity.entity()
                entity_seq = []
            if 'POINT' in tag:
                entity_tagger(instance, char, entity_seq, i, tag, 'POINT')
            elif 'PERIOD' in tag:
                entity_tagger(instance, char, entity_seq, i, tag, 'PERIOD')
            elif 'DURATION' in tag:
                entity_tagger(instance, char, entity_seq, i, tag, 'DURATION')
            elif 'FUZZY' in tag:
                entity_tagger(instance, char, entity_seq, i, tag, 'FUZZY')
            elif 'PROPE' in tag:
                entity_tagger(instance, char, entity_seq, i, tag, 'PROPE')
            if validate_entity(instance):
                list_entities.append(instance)
                instance = entity.entity()
                entity_seq = []
    return list_entities

def entity_tagger(entity, char, entity_seq, i, tag, type):
    if tag == 'B-'+type:
        entity.start = i
        entity_seq.append(char)
    # elif entity.type == type_dict[type]:
    elif tag == 'I-'+type:
        entity_seq.append(char)
    elif tag == 'S-'+type:
        entity.start = i
        entity.end = i + 1
        entity.entity = char
    elif tag == 'E-'+type:
        entity.type = type_dict[type]
        entity.end = i + 1
        entity_seq.append(char)
        entity.entity = ''.join(entity_seq)

def validate_entity(entity):
    if entity.entity and entity.end >= entity.start >=0:
        if len(entity.entity) == entity.end - entity.start:
            return True
    return False