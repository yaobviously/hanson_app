# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 19:32:07 2023

@author: yaobv
"""

import os
import openai
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

openai.organization = os.environ.get('OPENAI_ORG')
openai.api_key = os.environ.get('OPENAI_KEY')

def get_embedding(text, model="text-embedding-ada-002"):
    "simple function to get openai embeddings"
    return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']


if __name__ == '__main__':
    df = pd.read_csv('data/hanson_df.csv')
    blog_summaries = df['body'].values

    # creating the embeddings for the blog posts
    blog_embeddings = []

    for i, val in enumerate(blog_summaries):
        print(i, len(blog_summaries))
        embedding_ = get_embedding(val)
        blog_embeddings.append(embedding_)

    embeddings = np.array(blog_embeddings)
    np.save('data/blog_embeddings.npy', embeddings)

    # creating the embeddings for the grabby aliens summary
    grabby_df = pd.read_csv('data/grabby_df.csv')
    grabby_summaries = grabby_df['body'].values

    grabby_embeddings = []

    for summary in grabby_summaries:
        embedding_ = get_embedding(summary)
        grabby_embeddings.append(embedding_)

    grabby_embeddings = np.array(grabby_embeddings)
    np.save('data/grabby_aliens_embeddings.npy', grabby_embeddings)