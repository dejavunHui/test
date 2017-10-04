# -*- coding: utf-8 -*-
def average(*args):
    if len(args)==0:
        return 0.0
    sum=0.0
    for n in args:
        sum= sum+n
    return sum/len(args)

print(average())
print(average(1,2))
print(average(1,2,3,4,5))