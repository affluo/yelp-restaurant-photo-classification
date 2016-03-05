import pandas as pd
import numpy as np

# Loading data using pandas

# List of photos with the relevant business ID's
photos_biz = pd.read_csv('~/Projects/data/yelp/train_photo_to_biz_ids.csv')

# List of businesses with string indicating relevant tags
biz_label = pd.read_csv('~/Projects/data/yelp/train_photo_to_biz_ids.csv')
