"""
Below code is used to retrieve useful attributes for the recommender systems from raw product review and meta data.
Since the size of the raw data is too big for most personal computers, the data was retrived by chunk.

Raw data source:http://jmcauley.ucsd.edu/data/amazon/
Raw review data: reviews_Clothing_Shoes_and_Jewelry.json 2.71 GB
Raw meta data: meta_Clothing_Shoes_and_Jewelry.json 1.45 GB
"""

import pandas as pd
from datetime import datetime

##parse orignal data, select useful attributes by chunk
def parse(path,chunk):
  #one chunk contains up to 1000000 records
  with open(path, 'r') as g:
    counter = 0
    for l in g:
        counter += 1
        if counter <= 1000000 * (chunk-1):
            pass
        elif counter <= 1000000 * chunk:
            yield eval(l)
        else:
            break

def getDF(path,chunk):
  i = 0
  df = {}
  for d in parse(path,chunk):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')


def process_review_chunk(path, chunk):
    df = getDF(path,chunk)
    df['reviewTime'] = df['reviewTime'].map(lambda x: datetime.strptime(x, '%m %d, %Y'))
    #select a subset of df: drop unuseful columns and filter out only records with reviewTime later than 
    df.drop(['reviewerName','reviewText','summary','unixReviewTime','helpful'],axis = 1,inplace = True)
    df = df[df['reviewTime']>datetime(2012,1,1)]
    df = df.sort(['asin', 'reviewTime'], ascending=[1, 1])
    df.to_csv('review%d.csv' %chunk)

def process_meta_chunk(path,chunk):
    df = getDF(path,chunk)
    df = df.drop(['imUrl','related','salesRank'],axis = 1)
    df = df.sort('asin', ascending= True)
    df.to_csv('meta%d.csv' %chunk)

#for chunk in range(1,6):
#    process_review_chunk('reviews_Clothing_Shoes_and_Jewelry.json', chunk)

#for chunk in range(1,5):
#    process_meta_chunk('meta_Clothing_Shoes_and_Jewelry.json', chunk)


##conbine chunk data 
#combine meta data of all chunks
meta1 = pd.read_csv('meta1.csv')
meta2 = pd.read_csv('meta2.csv')
meta = pd.concat([meta1,meta2],axis = 0)
#select a random sample of 20% items 
meta_sample = meta.sample(n = 300000, replace = False)


#combine review data of all chunks
review1 = pd.read_csv('review1.csv')
review2 = pd.read_csv('review2.csv')
review3 = pd.read_csv('review3.csv')
review4 = pd.read_csv('review4.csv')
review5 = pd.read_csv('review5.csv')
review6 = pd.read_csv('review6.csv')
review = pd.concat([review1,review2,review3,review4,review5,review6],axis = 0)
#review.to_csv('review.csv')

#combine review data of all chunks
meta1 = pd.read_csv('meta1.csv')
meta2 = pd.read_csv('meta2.csv')
meta = pd.concat([meta1,meta2],axis = 0)
#meta.to_csv('meta.csv')

#merge review data with meta data base on asin 
review_meta = review.merge(meta, how = 'outer',on = 'asin')
review_meta_sample = review.merge(meta_sample, how = 'outer',on = 'asin')

#filter items and users based on a threshold of number of ratings 
df = pd. read_csv('sample_12kAsins')
