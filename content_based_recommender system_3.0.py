import pandas as pd
import numpy as np
import random 
import scipy as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics

##############load data##########################################
meta = pd.read_csv('meta.csv')
sample_rating = pd.read_csv('review_sample_not_treated_48305rows.csv')

item_sample = pd.DataFrame(sample_rating['asin'].unique(), columns = ['asin']).sample(3000)
meta = pd.merge(meta, item_sample, on = 'asin', how = 'inner')
sample_rating = pd.merge(sample_rating, item_sample, on = 'asin', how = 'inner') 

#inspect data sparsity 
meta[meta.categories.notnull()]   #1503384 rows  100%
meta[meta.description.notnull()]  #84781 rows   5.6%
meta[meta.title.notnull()]   #1502696 rows   99.9% 
meta[meta.price.notnull()]   #574882 rows   38.2% 
meta[meta.brand.notnull()]   #97407 rows   6.5%

###############Data transformation################################
#transform description and title through up to tri-gram 
def generate_tfidf(text_series):
    tf = TfidfVectorizer(analyzer='word',
                        ngram_range=(1, 3),
                        lowercase = True,
                        max_df = 0.3,
                        min_df = 0.01,
                        stop_words='english',
                        smooth_idf = True,
                        max_features = 50)                       
    tfidf = tf.fit_transform(text_series.values.astype('U'))                   
    tfidf = pd.DataFrame(data = tfidf.toarray(), index = meta['asin'],columns = tf.get_feature_names())
    return tfidf
                                                                                  
#tfidf_desc = generate_tfidf(meta['description'])
tfidf_title = generate_tfidf(meta['title'])


#combine categories, description and title matrixs and price
#price = pd.DataFrame(meta['price']).set_index(meta['asin'])
#item_profile = pd.concat([tfidf_cate, tfidf_desc, tfidf_title], axis = 1)

#generate user-item rating matrix 
rating_matrix = sample_rating.pivot(index='asin', columns='reviewerID', values='overall')
rating_matrix = rating_matrix.fillna(0) #replace rating for not rated items with 0, assume not rating items are disliked 


######################Generate training and testing set########################
def make_train(ratings, pct_test = 0.2):
    '''
    This function will take in the original user-item matrix and "mask" a percentage of the original ratings where a
    user-item interaction has taken place for use as a test set. The test set will contain all of the original ratings, 
    while the training set replaces the specified percentage of them with a zero in the original ratings matrix. 
    
    parameters: 
    
    ratings - the original ratings matrix from which you want to generate a train/test set. Test is just a complete
    copy of the original set. This is in the form of a sparse csr_matrix. 
    
    pct_test - The percentage of user-item interactions where an interaction took place that you want to mask in the 
    training set for later comparison to the test set, which contains all of the original ratings. 
    
    returns:
    
    training_set - The altered version of the original data with a certain percentage of the user-item pairs 
    that originally had interaction set back to zero.
    
    test_set - A copy of the original ratings matrix, unaltered, so it can be used to see how the rank order 
    compares with the actual interactions.
    
    user_inds - From the randomly selected user-item indices, which user rows were altered in the training data.
    This will be necessary later when evaluating the performance via AUC.
    '''
    rm = ratings.copy()
    rm[rm >= 3] = 1 # regards ratings equal or above three as like
    rm[rm < 3] = 0 # regard ratings below three as dislike  
    test_set = rm.copy() # Make a copy of the filled rating matrix set to be the test set. 
    training_set = rm.copy() # Make a copy of the original data we can alter as our training set. 
    nonzero_inds = training_set.nonzero() # Find the indices in the ratings data where an interaction exists
    nonzero_pairs = list(zip(nonzero_inds[0], nonzero_inds[1])) # Zip these pairs together of user,item index into list
    random.seed(0) # Set the random seed to zero for reproducibility
    num_samples = int(np.ceil(pct_test*len(nonzero_pairs))) # Round the number of samples needed to the nearest integer
    samples = random.sample(nonzero_pairs, num_samples) # Sample a random number of user-item pairs without replacement
    user_inds = [index[0] for index in samples] # Get the user row indices
    item_inds = [index[1] for index in samples] # Get the item column indices
    training_set[user_inds, item_inds] = 0 # Assign all of the randomly chosen user-item pairs to zero
    training_set.eliminate_zeros() # Get rid of zeros in sparse array storage after update to save space
    return training_set, test_set # Output the unique list of user rows that were altered  

rating_matrix_sp = sp.sparse.csr_matrix(rating_matrix.values)
product_train, product_test = make_train(rating_matrix_sp, pct_test = 0.2)

################Training the model#######################################
#generate top n recommendations
def train(item_profile, n):
    similar_items = {}
    cosine_similarities = linear_kernel(item_profile, item_profile)
    for idx, row in meta.iterrows():
        similar_indices = cosine_similarities[idx].argsort()[:-n-1:-1]
        for i in similar_indices:
            if meta['asin'][idx] not in similar_items:
                similar_items[meta['asin'][idx]] = [meta['asin'][i]]
            else:
                similar_items[meta['asin'][idx]] += [meta['asin'][i]]
    similar_items = pd.DataFrame.from_dict(similar_items, orient = 'index')
    return similar_items

recommendation = train(item_profile,3)


#####################Evaluation############################################
def auc_score(predictions, test):
    '''
    This simple function will output the area under the curve using sklearn's metrics. 
    
    parameters:
    
    - predictions: your prediction output
    
    - test: the actual target result you are comparing to
    
    returns:
    
    - AUC (area under the Receiver Operating Characterisic curve)
    '''
    fpr, tpr, thresholds = metrics.roc_curve(test, predictions)
    return metrics.auc(fpr, tpr)   

def calc_mean_auc(training_set, altered_users, predictions, test_set):
    '''
    This function will calculate the mean AUC by user for any user that had their user-item matrix altered. 
    
    parameters:
    
    training_set - The training set resulting from make_train, where a certain percentage of the original
    user/item interactions are reset to zero to hide them from the model 
    
    predictions - The matrix of your predicted ratings for each user/item pair as output from the implicit MF.
    These should be stored in a list, with user vectors as item zero and item vectors as item one. 
    
    altered_users - The indices of the users where at least one user/item pair was altered from make_train function
    
    test_set - The test set constucted earlier from make_train function
    
    
    
    returns:
    
    The mean AUC (area under the Receiver Operator Characteristic curve) of the test set only on user-item interactions
    there were originally zero to test ranking ability in addition to the most popular items as a benchmark.
    '''
    
    
    store_auc = [] # An empty list to store the AUC for each user that had an item removed from the training set
    popularity_auc = [] # To store popular AUC scores
    pop_items = np.array(test_set.sum(axis = 0)).reshape(-1) # Get sum of item iteractions to find most popular
    item_vecs = predictions[1]
    for user in altered_users: # Iterate through each user that had an item altered
        training_row = training_set[user,:].toarray().reshape(-1) # Get the training set row
        zero_inds = np.where(training_row == 0) # Find where the interaction had not yet occurred
        # Get the predicted values based on our user/item vectors
        user_vec = predictions[0][user,:]
        pred = user_vec.dot(item_vecs).toarray()[0,zero_inds].reshape(-1)
        # Get only the items that were originally zero
        # Select all ratings from the MF prediction for this user that originally had no iteraction
        actual = test_set[user,:].toarray()[0,zero_inds].reshape(-1) 
        # Select the binarized yes/no interaction pairs from the original full data
        # that align with the same pairs in training 
        pop = pop_items[zero_inds] # Get the item popularity for our chosen items
        store_auc.append(auc_score(pred, actual)) # Calculate AUC for the given user and store
        popularity_auc.append(auc_score(pop, actual)) # Calculate AUC using most popular and score
    # End users iteration
    
    return float('%.3f'%np.mean(store_auc)), float('%.3f'%np.mean(popularity_auc))  
   # Return the mean AUC rounded to three decimal places for both test and popularity benchmark
