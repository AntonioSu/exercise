# -*- coding: utf-8 -*-

import re
inputStr="hello python,ni hao c,zai jian python"
replaceStr=re.sub(r"hello (\w+),ni hao (\w+),zai jian \1","PHP",inputStr)
print replaceStr
#代码中的\1表示第一次匹配到的字符串,匹配成功，则替换字符串

inputStr="hello python,ni hao python,zai jian python"
replaceStr=re.sub(r"hello (\w+),ni hao (\w+),zai jian \2","PHP",inputStr)
print replaceStr
#代码中的\2表示第二次匹配到的字符串,匹配成功，则替换字符串
