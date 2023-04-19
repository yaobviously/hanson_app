# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 13:55:59 2023

@author: yaobv
"""
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer

def get_hanson_articles(query=None, top_k=5):
    """    
    a function to return the body of the top match and the blog title
    of the top 5 entries (for now, testing)
    """
    
    query_embedding = model.encode(query)
    
    d, i =  embeddings_idx.search(np.array([query_embedding]), top_k)

    post_bodies = [posts[x] for x in i[0]]
    post_titles = [titles[x] for x in i[0]]
    links_ = [links[x] for x in i[0]]
    
    return post_bodies[0] + post_bodies[1], links_[0]

def hanson_chat_test(query_test=None):    
    
    posts, link = get_hanson_articles(query=query_test, top_k=2)
    prompt = messages.copy()
    
    prompt.append({"role" : "user", "content" : f"I would like you to consider the following before answering the question: '{posts}'. Note that if the foregoing doesn't seem relevant, please inform the user that Hanson doesnt address that on his blog. Stick to what Hanson would say. {query_test}"})
    
    response = (
        openai
        .ChatCompletion
        .create(
            model="gpt-3.5-turbo",
            max_tokens=300,
            messages=prompt)
        )
    print(response.usage.completion_tokens)

    generated_text = response.choices[0].message.content
    
    messages.append({"role" : "user", "content" : f'{query_test}'})
    messages.append({"role" : "assistant", "content" : generated_text})
    
    return generated_text, link