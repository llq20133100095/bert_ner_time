#coding=utf-8
'''
call example of time_duration_apm.json

created by: gagapeng,
created on: 2019-01-16-16:00,
modified on: 2019-01-16-16:00,
'''

import json 

time_data_file = open("resources/time_duration_apm.json",mode='r')
time_dict=json.loads(time_data_file.read())
time_data_file.close()

print(time_dict[u"寒冬"]["start"])
print(time_dict[u"寒冬"]["end"])


