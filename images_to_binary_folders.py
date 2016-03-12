import pandas as pd
import numpy as np

# Loading data using pandas

# List of photos with the relevant business ID's
photos_biz = pd.read_csv('~/Projects/data/yelp/train_photo_to_biz_ids.csv')

# List of businesses with string indicating relevant tags
biz_label = pd.read_csv('~/Projects/data/yelp/train.csv')

# Create a dictionary with photo_id as the key and labels as the value
d = {}
for j in photos_biz.index:
    d[photos_biz['photo_id'][j]] = biz_label['labels'][biz_label['business_id'][biz_label['business_id'] == photos_biz['business_id'][j]].index[0]]
        
        