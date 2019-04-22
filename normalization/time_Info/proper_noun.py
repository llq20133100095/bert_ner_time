# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 16:23:47 2019

@author: leolqli
"""
import re
import sqlite3
import json

class Proper_noun:
    
    def __init__(self):
        self.pattern_year = u'.*?([0-9]{4}).*?'
        self.DB_FILE = './db/lunarcal.sqlite'
        self.file_proper_noun = "../resources/proper_noun.json"
        #proper_noun_dict
        self.proper_noun_dict = json.load(open(self.file_proper_noun, "r"))
        
        
    def query_db(self, query, args=(), one=False):
        ''' 
        Function:
            wrap the db query, fetch into one step 
        Parameters:
            1.query: sql
            2.args: have start, end, entity
        '''
        conn = sqlite3.connect(self.DB_FILE)
        conn.row_factory = sqlite3.Row
        db = conn.cursor()
        cur = db.execute(query, args)
        rv = cur.fetchall()
        cur.close()
        return (rv[0] if rv else None) if one else rv

    def byteify(self, input):
        """
        Function:
            change the unicode of dictionary to the string
        """
        if isinstance(input, dict):
            return {self.byteify(key): self.byteify(value) for key, value in input.iteritems()}
        elif isinstance(input, list):
            return [self.byteify(element) for element in input]
        elif isinstance(input, unicode):
            return input.encode('utf-8')
        else:
            return input
        
    def update_holiday(self, dict_key, dict_content):
        """
        Function:
            update the holiday name in proper_noun.json
        Parameters:
            1.dict_key: str, holiday alias
            2.dict_content: str, holiday true name
        """
        read_proper_noun = open(self.file_proper_noun, "r")
        origin_proper_dict = json.load(read_proper_noun)
        read_proper_noun.close()
        
        origin_proper_dict = self.byteify(origin_proper_dict)
        origin_proper_dict[dict_key] = dict_content
        
        save_proper_noun = open(self.file_proper_noun, "w")
        json.dump(origin_proper_dict, save_proper_noun, ensure_ascii=False, indent=2)
        save_proper_noun.close()
      
    def solar_parse(self, entity, basetime):
        """
        Function:
            1.select the time_name in db file "lunarcal.sqlite"
        Parameters:
            1.entity; str
            2.basetime: str
        """
        year = re.match(self.pattern_year, basetime)
        year = year.group(1).decode("utf-8")
        
        start = year + '-01-01'
        end_year = str(int(year) + 1)
        end = end_year + '-12-31'
        
#        if(u'节' in entity and entity != u'春节'):
#            entity = entity[:-1]
        
        if(entity in self.proper_noun_dict.keys()):
            entity = self.proper_noun_dict[entity]
        
            sql = ('select * from ical '
                   'where date>=? and date<=? and (holiday = ? or jieqi = ?) order by date')
            rows = self.query_db(sql, (start, end, entity, entity))
            
            date = ""
            for r in rows:
                date = r['date']
                for result in r:
                    print result
            return date
        
        else:
            return None

if __name__ == "__main__":
    proper_noun = Proper_noun()
    entity = u'春节'
    basetime = u'2019-02-14 12:00:00'
    
    date_pre, date2_next = proper_noun.solar_parse(entity, basetime)
#    proper_noun.update_holiday("春节", "春节")
    
#    conn = sqlite3.connect(proper_noun.DB_FILE)
#    db = conn.cursor()
#    sql = 'update ical set holiday=? where holiday=?'
#    db.execute(sql, (u'七夕', u'七夕情人节'))    
#    conn.commit()
    
    
    """2.save the proper noun in json"""
#    a = {"腊八节": "腊八", "腊八": "腊八",
#         "元宵节": "元宵", "元宵": "元宵",
#         "寒食节": "寒食", "寒食": "寒食",
#         "端午节": "端午", "端午": "端午",
#         "七夕节": "七夕", "七夕": "七夕", "七夕情人节": "七夕",
#         "中元节": "中元", "中元": "中元",
#         "中秋节": "中秋", "中秋": "中秋",
#         "重阳节": "重阳", "重阳": "重阳",
#         "下元节": "下元", "下元": "下元"}
#    file_proper_noun = open("../resources/proper_noun.json", "w")
#    json.dump(a, file_proper_noun, ensure_ascii=False, indent=2)
#    file_proper_noun.close()
    
    