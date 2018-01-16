# for Python 3
# Day of week examples

import datetime

dow = {0: 'Monday', 1: 'Tuesday', 2:'Wednesday', 3:'Thursday',
4:'Friday', 5:'Saturday', 6:'Sunday'}

dow2 = {0: [1,0,0,0,0], 1:[0,1,0,0,0], 2:[0,0,1,0,0], 3:[0,0,0,1,0],
4:[0,0,0,0,1]}

d = datetime.date(2018, 1, 16)

wd = d.weekday()

print(wd)
print(dow[wd])
print(dow2[wd])

