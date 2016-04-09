import pandas as pd

#First need to create 9 text files representing label probabilites for each photo
import pandas as pd
import numpy as np

# Loading data using pandas

# Dataframe, photos with the relevant business ID's
photos_biz = pd.read_csv('~/Projects/data/yelp/train_photo_to_biz_ids.csv')

# List of businesses with string indicating relevant tags
biz_label = pd.read_csv('~/Projects/data/yelp/train.csv')

# Replace NaN in labels with an empty string
biz_label['labels']=biz_label['labels'].replace(np.nan,' ', regex=True)

# Create a dictionary with photo_id as the key and labels as the value
photos_label = {}
for j in photos_biz.index:
    photos_label[photos_biz['photo_id'][j]] = biz_label['labels'][biz_label['business_id'][biz_label['business_id'] == photos_biz['business_id'][j]].index[0]]


# THis was to create the random photo 
# From the dictionary, create text files photo_id, prob_label
#text_file_list = ['0.txt','1.txt','2.txt','3.txt','4.txt','5.txt','6.txt','7.txt','8.txt']
#
#
#for j in range (9):
#    text_file = open("/Users/robsalz/Projects/yelp/images_binary_probabilities/"+text_file_list[j], "w")
#    text_file.write ('photo_id, probability \n')    
#    for i in photos_label:
#        s = str(i) + ', ' + str(np.random.random()) + '\n'
#        text_file.write (s)
#    text_file.close()

prob0 = pd.read_csv('~/Projects/yelp/images_binary_probabilities/0.txt')

# Create a dictionary of photo_id keys and business_id values
photos_biz_dict = {}
for i in photos_biz.index:
    photos_biz_dict[photos_biz['photo_id'][i]]=photos_biz['business_id'][i]
  

#Takes the probability folder and finds the business id for each photo (CAUTION!!!! SLOW)  
#prob0['business_id']=0

#for i in prob0.index:
#    prob0['business_id'][i] = photos_biz_dict[prob0['photo_id'][i]]


              
# Making a list of lists: 
# In each sub list is the bussiness probabilities for a given label
###############################################################################
#Sort by business, and reindex the dataframe
prob0.sort(columns = 'business_id')
prob0 = prob0.reset_index(drop = true)

# Create an empty list to store the probabilities for each business in      
prob_list = []
for i in biz_label.index:
    prob_list.append([])
    
# Iterate through sorted dataframe, and apend probability to each businesses empty list    
biz = 0
for i in prob0.index:
    prob_list[biz].append(prob0[' probability '][i])
    if i = len(prob0)-1:
        break
    if prob0['business_id'][i] != prob0['business_id'][i+1]:
        biz = biz + 1;
    
# Extracting business statistics from list of probabilities: 
###############################################################################    

#Create a list of numpy arrays from the list of lists:
prob_array_list = []
for biz_probs in prob_list:
    prob_array_list.append(np.asarray(biz_probs))
    
# Want to find the maximum number of photos to normalize the num_photos feature
max_photos = max(len(prob) for prob in prob_list)

# Create a list of all businesses stats, individual stats save in a vector
biz_stats_list = []
for biz_probs in prob_array_list:
    biz_max = np.amax(biz_probs)
    biz_min = np.amin(biz_probs)
    biz_mean = np.mean(biz_probs)
    biz_std = np.std(biz_probs)
    biz_no_photos = float(len(biz_probs))/max_photos
    biz_stats = np.array([biz_max,biz_min,biz_mean,biz_std,biz_no_photos])
    biz_stats_list.append(biz_stats)
    
# Hot Encoding the labels:
###############################################################################
#Sort this by business and reindex (These guys or cunts for giving csv out of order)
biz_label = biz_label.sort(columns = 'business_id')
biz_label.reset_index(drop=True)

def hot_encode (label, num_classes=9):
    '''Takes string of form '0 1 9' and returns a hot encoding'''
    import numpy as np
    hot_encoding = np.zeros(num_classes)
    
    label = label.replace(" ", "")
    for l in label:
        l = int(l)
        hot_encoding[l] = 1
    return hot_encoding
    
hot_labels = []
for label in biz_label['labels']:
    hot_label = hot_encode(label)
    hot_labels.append(hot_label)