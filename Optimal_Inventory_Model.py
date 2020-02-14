# -*- coding: utf-8 -*-
"""
Created on Mon Apr 08 21:22:34 2019

@author: win8
"""
"""
#import libraries
import math

#Creating base variables
N = int(input("Please enter number of locations:"))

#Conditions for base variables

N >= 1

#Prevention for errors to prevent users from accidentally or intentionally typing 
#wrong required in-puts (1 instead of “one”), a combination of infinite loop and 
#try/except function was created.

#Calculating EBOs

num = 1 #item number
num2 = 2

m = float(input("Please enter the average weekly demand for item"+ str(num)+":"))
print m

length = 100 #desired number of stock levels
"""

import math
#Creating base variables

while True:
    N = raw_input('Please enter your fleet size: ')
    try:
        N = int(N)
        N >= 1
        break
    except:
        print 'Wrong input!!! Please type only positive integer (-_-)'

while True:
    numitems = raw_input('Please enter number of critical items in one unit (It should be greater than 1): ')
    try:
        numitems = int(numitems)
        numitems > 1
        break
    except:
        print 'Wrong input!!! Please type only positive integer (-_-)'

while True:
    length = raw_input('Please enter number of stock level (s) taken into consideration for each critical item: ')
    try:
        length = int(length)
        length >= 1
        break
    except:
        print 'Wrong input!!! Please type only positive integer (-_-)'

#Creating a list of EBOs
while True:
    Prac = 0
    t = list()
    for num in range(1,numitems+1):
        while True:
            m = raw_input('Please enter the average annual demand for item ' + str(num) + ': ')
            try:
                m = float(m)
                m >= 0
                break
            except:
                print 'Wrong input!!! Please type only positive real numbers (-_-)'
        while True:
            T = raw_input('Please enter the average repair time in years for item ' + str(num) + ': ')
            try:
                T = float(T)
                T >= 0
                break
            except:
                print 'Wrong input!!! Please type only positive real numbers (-_-)'

        apl = m*T
        for x in range(length):
            Pr = (apl**x)*math.exp(-apl)/math.factorial(x)
            Prac = Prac + Pr
            ebo = apl*Pr + (apl-x) * (1-Prac)
            t.append(ebo)
        Prac = 0
    t = [t[p:p+length] for p in range(0,len(t),length)]
    #print t
    countitems = 1
    count1 = 0
    count2 = 0
    while True:
        for p in t[count1]:
            print ('Item ' + str(countitems) + ' EBO at s = ' + str(count2) + ' : ' + str(p))
            count2 = count2 + 1
            if count2 > (length-1): break
        if countitems >= numitems or count1 >= (numitems-1): break
        else:
            countitems = countitems + 1
            count2 = 0
            count1 = count1+1
    print 'Please check the raw data once again !!!'
    print 'Type (Yes) in case you want to continue, (No) to reinput the data.'
    while True:
        ques = raw_input('Continue or not ? ')
        if ques == 'Yes': break
        elif ques == 'No': break
        else:
            print 'Please answer only (Yes) or (No) !!!'
    #print t
    if ques == 'Yes': break
#Creating a list of unit cost for each items
while True:
    costdata = list()
    for p in range(1,numitems+1):
        while True:
            uc = raw_input('Please enter unit cost for item ' + str(p) + ': ')
            try:
                uc = float(uc)
                costdata.append(uc)
                break
            except:
                print 'Wrong input!!! Please enter positive integer (-_-)'
    print 'Please check the unit cost data once again !!!'
    print 'Type (Yes) in case you want to continue, (No) to reinput the data.'
    while True:
        ques = raw_input('Continue or not ? ')
        if ques == 'Yes': break
        elif ques == 'No': break
        else:
            print 'Please answer only (Yes) or (No) !!!'
    #print costdata
    if ques == 'Yes': break
#Creating combination of stock level between items
temp = range(length)*numitems
names = [temp[p:p+length] for p in range(0,len(temp),length)]
#print names
while True:
    if len(names) <= 1: break
    else:
        nametemp = list()
        for n1 in names[0]:
            for n2 in names[1]:
                name = str(n1) + "&" + str(n2)
                nametemp.append(name)
        names[0:2] = []
        names.insert(0,nametemp)
names = names[0]
#print names
#Creating a cost list for each stock level
i = 0
inp = range(length)*numitems
inp = [inp[p:p+length] for p in range(0,len(temp),length)]
temp = list()
while i < len(inp):
    for c in costdata:
        for t1 in inp[i]:
            cost = c*t1
            temp.append(cost)
        i = i + 1
costlist = [temp[p:p+length] for p in range(0,len(temp),length)]
#print costlist
#Making final cost list for each comabination
while True:
    if len(costlist) <= 1: break
    else:
        temp = list()
        for c1 in costlist[0]:
            for c2 in costlist[1]:
                sum = c1 + c2
                temp.append(sum)
        costlist[0:2] = []
        costlist.insert(0,temp)
costlist = costlist[0]
#print costlist
#Creating a list for occurences
occlist = list()
while True:
    for num in range(1,numitems+1):
        mem = raw_input('Please enter occurences for item ' + str(num) + ': ')
        if mem == 'quit': quit()
        try:
            mem = float(mem)
            occlist.append(mem)
        except:
            print 'Wrong input!!! Please type again (-.-)'
    print 'Please check the occurences data once again !!!'
    print 'Type (Yes) in case you want to continue, (No) to reinput the data.'

    while True:
        ques = raw_input('Continue or not ? ')
        if ques == 'Yes': break
        elif ques == 'No': break
        else:
            print 'Please answer only (Yes) or (No) !!!'
    if ques == 'Yes': break
#print occlist
#Creating a list for system availability
temp1 = list()
ran = range(len(t))
Z = 0
#print i
for p in ran:
    for p1 in t[p]:
        ex = pow(1-(p1/(N*occlist[Z])),occlist[Z])
        temp1.append(ex)
    Z = Z + 1
temp1 = [temp1[p:p+length] for p in range(0,len(temp1),length)]
#print temp1
while True:
    if len(temp1) <= 1: break
    else:
        temp2 = list()
        for p2 in temp1[0]:
            for p3 in temp1[1]:
                ava100 = p2*p3
                temp2.append(ava100)
        temp1[0:2] = []
        temp1.insert(0,temp2)
temp1 = temp1[0]
#print temp1
while True:
    if len(t) <= 1: break
    else:
        temp = list()
        for t1 in t[0]:
            for t2 in t[1]:
                sum = t1 + t2
                temp.append(sum)
        t[0:2] = []
        t.insert(0,temp)
t = t[0]
#print t

#Creating dictionaries
costdict = dict()
ebodict = dict()
avadict = dict()
optdata = dict()
pos1 = 0
pos2 = 0
pos3 = 0
for d in names:
    ebodict[d] = t[pos1]
    pos1 = pos1 + 1
#print ebodict
for d in names:
    costdict[d] = costlist[pos2]
    pos2 = pos2 + 1
#print costdict
for d in names:
    avadict[d] = temp1[pos3]
    pos3 = pos3 + 1
#print avadict
#Finding optimal inventory policy
while True:
    print 'Type (a) for constraint of availability and (c) for cost'

    quesf = raw_input('Would you like to choose constraint for availability or cost? ')
    if quesf == 'c':
        while True:
            ltdcost = raw_input('Please enter maximum allowable cost for spares: ')
            if float(ltdcost) < min(costdict.itervalues()):
                print 'The cost is too small, please enter at least',min(costdict.itervalues())
                continue
            try:
                ltdcost = float(ltdcost)
                optdata1 = dict()
                for key,value in costdict.iteritems():
                    if value > ltdcost: continue
                    else:
                        for key1,value1 in avadict.iteritems():
                            if key != key1: continue
                            else:
                                optdata1[key] = value1
                #print optdata
                com = max(optdata1, key=optdata1.get)
                syscost = costdict[com]
                syscost = round(syscost,3)
                sysebo = ebodict[com]
                sysebo = round(sysebo,3)
                sysava = avadict[com]

                sysava = round(sysava*100,3)
                com = com.split('&')
                num = 1
                print 'Optimal inventory policy: '
                for p in com:
                    print 'Item',num,':',p
                    num = num + 1
                print 'System EBO(s):',sysebo
                print ('System Availability: ' + str(sysava) + '%')
                print 'System Cost:',syscost
                while True:
                    ques1 = raw_input('Would like to try another allowable cost (Type Yes or No)? ')
                    if ques1 == 'Yes': break
                    elif ques1 == 'No': break
                    else:
                        print 'Wrong input!!! Please type only Yes or No (-_-)'
            except:
                print 'Wrong input!!! Please type only positive integer (-_-)'
            if ques1 == 'No': break
    elif quesf == 'a':
        while True:
            ltdava = raw_input('Please enter minimum allowable availability for the fleet (float Number e.g. 0.98): ')
            if float(ltdava) < min(avadict.itervalues()):
                print 'The availability is too small, please enter at least',min(avadict.itervalues())
                continue
            try:
                ltdava = float(ltdava)
                optdata2 = dict()
                for key,value in avadict.iteritems():
                    if value < ltdava: continue
                    else:
                        for key1,value1 in costdict.iteritems():
                            if key != key1: continue
                            else:
                                optdata2[key] = value1
                #print optdata
                com = min(optdata2, key=optdata2.get)
                syscost = costdict[com]
                syscost = round(syscost,3)
                sysebo = ebodict[com]
                sysebo = round(sysebo,3)
                sysava = avadict[com]
                sysava = round(sysava*100,3)
                com = com.split('&')
                num = 1
                print 'Optimal inventory policy: '
                for p in com:
                    print 'Item',num,':',p
                    num = num + 1
                print 'System EBO(s):',sysebo
                print ('System Availability: ' + str(sysava) + '%')
                print 'System Cost:',syscost
                while True:
                    ques1 = raw_input('Would like to try another allowable availability (Type Yes or No)? ')
                    if ques1 == 'Yes': break
                    elif ques1 == 'No': break
                    else:
                        print 'Wrong input!!! Please type only Yes or No (-_-)'
            except:
                print 'Wrong input!!! Please type only positive integer (-_-)'
            if ques1 == 'No': break
    else:
        print 'Wrong input!!! Please type only (a) or (c) (-_-))'
        continue
    while True:
        quesc = raw_input('Would you like to change the constraint? ')
        if quesc == 'Yes':
            break
        elif quesc == 'No': break
        else:
            print 'Wrong input!!! Please type only Yes or No (-_-)'
    if quesc == 'No': break