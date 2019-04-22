# bert-ner-time[leolqli]
-------------------------------------
## 一、修改的地方
    1."./utils/digitconv.py": 
          u'天': 7,
          u'日': 7,
          }
    
    2."./normalization/CommonParser.py":
    (1)新增加"timeLength" parameter: list[年、月、周、日、时、分、秒]， 用来表示时间的长度。其中"time_duration"和"frequency"输出需要用到。
    
    (2)新增加"is_approximation" parameter: 用来判断该实体是否为概数"前后"，"左右"。
-------------------------------------

## 二、新增的程序

### 1.分析报告详情在腾讯文档:[Time_analyse分析报告](https://docs.qq.com/doc/DU1hxQ1JiQXB4WGNp)

### 2.time_period type
    (1)"./normalization/TimePeriodProcess.py":处理"time_period" type，such  as "week" and "season"
    
    (2)"./normalization/time_Info/week.py":处理"week",利用pattern方法
    
    (3)"./normalization/time_Info/season.py":处理"season",利用pattern方法

### 3.time_duration type
    (1)"./normalization/TimeDurationProcess.py":处理"time_duration" type，such as "70个月", output the "timeLength" list[年、月、周、日、时、分、秒]
    
    (2)"./normalization/time_Info/duration.py":处理"time_duration",利用pattern方法

### 4.frequency：时间频率集
    (1)"./normalization/TimeFrequencyProcess.py":处理"time_frequency" type，such as "每天...", output the "timeLength" list[年、月、周、日、时、分、秒]
    
    (2)"./normalization/time_Info/frequency.py":处理"time_frequency",利用pattern方法

### 5.approximation: 判断是否为概数
    (1)"./normalization/time_Info/approximation.py":处理"approximation"类型， samples like "7天左右". The output is the  "is_approximation", where the "True" is the approximation type.

### 6.time_period: 综合处理
    (1)"./normalization/TimePeriodSplit.py": 处理"到|至"的时间段, 其中它调用了week.py和TimePeriodAround.py
	
		1)处理suffix_time中缺失了前面时间的：2018年10月-11月
		
		2)处理prefix_time中缺失了后面时间的：10-11月、2014年1月17至1月24日
	
	(2)"./normalization/TimePeriodAround.py"
		3)处理周的时间：前一周、前两个礼拜
		
		4)处理带有“前”、“后”、“未来”、“明”、“后期”、“前后”、“里”、“开始”、“上”、“下”
		
			前缀：前、后、未来、明、上、昨、今、去
			数量词
			时间词语：世纪、年、月、日|天、apm、点|时、分|分钟|刻、秒
			
			特殊apm词语:仅仅只有apm词语的
			
			数量词
			时间词语：世纪、年、月、日|天、apm、点|时、分|分钟|刻、秒
			后缀词语：里、内、后期
			
		5)处理带有年代和世纪的时间
		
		6)处理带有季度的词语
		

### 7.proper_noun: 专有名词处理
    （1）代码文件："./normalization/time_Info/proper_noun.py"
    
    （2）处理传统的节假日：调用香港从1901-2100之间的旧历，然后存到数据库中进行查询。其数据在"./normalization/time_Info/db"
    
    （3）处理传统的节气

## 三、数据存放

### 1.time_period的数据

- './normalization/leolqli_data/time_period_reach.csv':时间段包含词语“到”
- './normalization/leolqli_data/time_period_kaifa_predict.csv':开发集
- './normalization/leolqli_data/time_period_finalcase50.csv':测试集共63条

## 四、测试集测试和错误分析

### 1.time_period的测试
    
    （1）.在测试集上，准确率达到82.5%

    （2）.错误case分析

        1）apm字典中没有覆盖完全，比如：明早

        2）时间点中没有识别到特殊的词语：现在为止，２００６年８月

        3）周和季度的识别没有覆盖到一些词语：过去，年和季度