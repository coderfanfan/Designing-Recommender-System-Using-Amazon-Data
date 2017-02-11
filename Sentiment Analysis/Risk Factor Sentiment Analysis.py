import re

##load LM words 
#load LM positive words 
poswords = list()
with open('pos_words.txt','r') as f:
    for line in f:
        line = line.lower()
        poswords += line.strip().split('\n')
        

#load LM negative words 
negwords = list()
with open('neg_words.txt','r') as f:
    for line in f:
        line = line.lower()
        negwords += line.strip().split('\n')


##Expedia Risk Factor Sentiment 

#Expedia Risk Factor total words
ExpeRF = list()
with open('Expedia 2015 Risk Factor.txt','r') as f:
    for line in f:
        line = line.lower()
        ExpeRF += re.findall(r"[a-z]+",line)
print 'There are {} words in the Expedia Risk Factor text file'.format(len(ExpeRF)) 

#calculate number of positive and negative words in Expedia Risk Factor
ExpeRFpos = 0
ExpeRFneg = 0
for word in ExpeRF:
    if word in poswords:
        ExpeRFpos += 1
    elif word in negwords:
        ExpeRFneg += 1

ExpeRFsent = (float(ExpeRFpos) - float(ExpeRFneg))/len(ExpeRF)
print 'Expedia Risk Factor text sentiment is {:.4f}'.format(ExpeRFsent)


##Priceline Risk Factor Sentiment 

#Priceline Risk Factor total words
PclnRF = list()
with open('Priceline 2015 Risk Factor.txt','r') as f:
    for line in f:
        line = line.lower()
        PclnRF += re.findall(r"[a-z]+",line)
print 'There are {} words in the Priceline Risk Factor text file'.format(len(PclnRF)) 

#calculate number of positive and negative words in Priceline Risk Factor
PclnRFpos = 0
PclnRFneg = 0
for word in PclnRF:
    if word in poswords:
        PclnRFpos += 1
    elif word in negwords:
        PclnRFneg += 1

PclnRFsent = (float(PclnRFpos) - float(PclnRFneg))/len(PclnRF)
print 'Priceline Risk Factor text sentiment is {:.4f}'.format(PclnRFsent)


##TripAdvisor Risk Factor Sentiment 

#TripAdvisor Risk Factor total words
TripRF = list()
with open('TripAdvisor 2015 Risk Factor.txt','r') as f:
    for line in f:
        line = line.lower()
        TripRF += re.findall(r"[a-z]+",line)
print 'There are {} words in the TripAdvisor Risk Factor text file'.format(len(TripRF)) 

#calculate number of positive and negative words in TripAdvisor Risk Factor
TripRFpos = 0
TripRFneg = 0
for word in TripRF:
    if word in poswords:
        TripRFpos += 1
    elif word in negwords:
        TripRFneg += 1

TripRFsent = (float(TripRFpos) - float(TripRFneg))/len(TripRF)
print 'TripAdvisor Risk Factor text sentiment is {:.4f}'.format(TripRFsent)