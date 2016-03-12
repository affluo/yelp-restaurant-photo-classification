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
    
# From the dictionary, create text files photo_id.jpg and 0 or 1 for each class
text_file_list = ['0.txt','1.txt','2.txt','3.txt','4.txt','5.txt','6.txt','7.txt','8.txt']

for j in range (9):
    text_file = open(text_file_list[j], "w")
    for i in photos_label:
       if str(j) in photos_label[i]:
           s = str(i) + '.jpg 1 \n'
           text_file.write (s)
       else:
           s = str(i) + '.jpg 0 \n'
           text_file.write (s)
    text_file.close()
       
     