# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 12:14:31 2023

@author: yaobv
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# loading the pretrained transformer
model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')

# loading hanson df
df = pd.read_csv('./hanson/hanson_df.csv')

# extracting the body values
headings = df['title'].values
bodies = df['body'].values
links = df['url'].values

#encoding posts
encoded_posts = model.encode(bodies)

# saving the encodings so we dont have to do that again: ~45 min
np.save('./hanson/hanson_blog_embeddings.npy', encoded_posts)