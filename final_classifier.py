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


#Writing a dummy csv file
# Create a dictionary with photo_id as the key and labels as the value
#photos_label = {}
#for j in photos_biz.index:
#    photos_label[photos_biz['photo_id'][j]] = biz_label['labels'][biz_label['business_id'][biz_label['business_id'] == photos_biz['business_id'][j]].index[0]]
#for j in range (9):
#    text_file = open("/Users/robsalz/Projects/yelp/"+'prob.csv', "w")
#    text_file.write ('photo_id,prob0,prob1,prob2,prob3,prob4,prob5,prob6,prob7,prob8\n')    
#    for i in photos_label:
#        s = str(i) + ', ' + str(np.random.random())+ ', ' + str(np.random.random())+ ', ' + str(np.random.random())+ ', ' + str(np.random.random())+ ', ' + str(np.random.random())+ ', ' + str(np.random.random())+ ', ' + str(np.random.random())+ ', ' + str(np.random.random())+ ', ' + str(np.random.random())+ '\n'
#        text_file.write (s)
#    text_file.close()

#Read in the CSV to prob datafram
prob = pd.read_csv('~/Projects/yelp/prob.csv')
prob = prob.sort('photo_id') #Sort by photo
prob = prob.reset_index(drop=True)

#Append business id to prob datafram
prob_biz = photos_biz.sort(columns = 'photo_id') #Sort by photo first
prob_biz = prob_biz.reset_index(drop=True)  #Re - index the dataframe
prob['business_id']= prob_biz['business_id']

#Sort by business, and reindex the dataframe
prob = prob.sort(columns = 'business_id')
prob = prob.reset_index(drop = True)

#All photo probabilities to business stats vector 

# Iterate through each class
# Will work with one class at store the stats in all_stats
all_stats = []
for cls in xrange (9):              
    # Create an empty list to store the class probabilities for each business in      
    prob_list = []
    for i in biz_label.index:
        prob_list.append([])
            
    # Iterate through sorted dataframe, and apend probability to each businesses empty list    
    biz = 0
    for i in prob.index:
        prob_list[biz].append(prob['prob'+str(cls)][i])
        if i == len(prob)-1:
            break
        if prob['business_id'][i] != prob['business_id'][i+1]:
            biz = biz + 1;
        
    # Extracting business statistics from list of probabilities:         
    # Create a list of numpy arrays from the list of lists:
    prob_array_list = []
    for biz_probs in prob_list:
        prob_array_list.append(np.asarray(biz_probs))
        
    # Want to find the maximum number of photos to normalize the num_photos feature
    max_photos = max(len(prob) for prob in prob_list)
    
    # Create a list of all businesses stats, individual stats saved in a vector
    biz_stats_list = []
    for biz_probs in prob_array_list:
        biz_max = np.amax(biz_probs)
        biz_min = np.amin(biz_probs)
        biz_mean = np.mean(biz_probs)
        biz_std = np.std(biz_probs)
        biz_no_photos = float(len(biz_probs))/max_photos
        biz_stats = np.array([biz_max,biz_min,biz_mean,biz_std,biz_no_photos])
        biz_stats_list.append(biz_stats)
    all_stats.append(biz_stats_list)

#Make input vecotrs out of the lists of stats
input_vectors = []

# Iterate through businesses and make each vector
for i in biz_label.index:
    input_vectors.append(np.concatenate((all_stats[0][i], all_stats[1][i], all_stats[2][i], all_stats[3][i], all_stats[4][i], all_stats[5][i], all_stats[6][i],all_stats[7][i],all_stats[8][i])))
#Make the list an array
input_vectors_asarray = np.asarray(input_vectors)

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

#Hot labels as an array
hot_labels_asarray = np.asarray(hot_labels)