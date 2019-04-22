#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/28 22:48
# @Author  : honeyding
# @File    : config.py
# @Software: PyCharm

import tensorflow as tf
import json
import six

class NerConfig(object):
    def __init__(self,
                 word2id_data="./resources/word2id.pkl",
                 train_data="./resources/time_train_200.txt",
                 test_data="./resources/time_test_200.txt",
                 batch_size=64,
                 epoch=40,
                 hidden_dim=300,
                 optimizer="Adam",
                 lr=0.001,
                 clip=5.0,
                 dropout=0.5,
                 update_embedding=True,
                 embedding_dim=300,
                 shuffle=True,
                 output="./output/"):

        self.word2id_data = word2id_data
        self.train_data = train_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.epoch = epoch
        self.hidden_dim = hidden_dim
        self.optimizer = optimizer
        self.lr = lr
        self.clip = clip
        self.dropout = dropout
        self.update_embedding = update_embedding
        self.embedding_dim = embedding_dim
        self.shuffle = shuffle
        self.output = output

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = NerConfig()
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with tf.gfile.GFile(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))