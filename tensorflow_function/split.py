#coding=utf-8
string = "www.gziscas.com.cn"

#1.以'.'为分隔符
print(string.split('.'))
#['www', 'gziscas', 'com', 'cn']

#2.分割两次
print(string.split('.',2))
#['www', 'gziscas', 'com.cn']

#3.分割两次，并取序列为1的项
print string.split('.',2)[1]
#gziscas
