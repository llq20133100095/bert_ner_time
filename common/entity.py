#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/02 22:20
# @Author  : honeyding
# @File    : entity.py
# @Software: PyCharm


class entity(object):

    def __init__(self):
        self.entity = None
        self.start = -1
        self.end = -1
        self.type = None
        self.abs_rel = None
        self.is_refer = False
        self.is_freq = False

