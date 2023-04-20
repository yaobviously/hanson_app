# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 19:12:18 2023

@author: yaobv
"""
import os
import pandas as pd
import numpy as np
import openai

import streamlit as st
from streamlit_chat import message

from dotenv import load_dotenv

load_dotenv()

# setting org and key
openai.organization = os.environ.get('OPENAI_ORG')
openai.api_key = os.environ.get('OPENAI_KEY')

# loading the dataframe and the embeddings array
df = pd.read_csv('data/hanson_df.csv')
embeddings = np.load('data/openai_hanson_embeddings.npy')

# extracting the dataframe values
posts = df['body'].values
titles = df['title'].values
links = df['url'].values

def get_embedding(text, model="text-embedding-ada-002"):
    return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']


def get_hanson_articles(question=None, top_k=5):
    """    
    a function to return the body of the top match and the blog title
    of the top 5 entries (for now, testing)
    """
    
    query_embedding = get_embedding(str(question))
    
    similarities = embeddings.dot(query_embedding)
    indices = np.argsort(-similarities)

    post_bodies = [posts[x] for x in indices]
    post_titles = [titles[x] for x in indices]
    links_ = [links[x] for x in indices]
    
    return post_bodies[0] + post_bodies[1], links_[:2], post_titles[:2]

get_hanson_articles(posts[55])

# defining page header
st.set_page_config(page_title="What Would Hanson Say?", page_icon=":robot_face:")
st.markdown("<h1 style='text-align: center;'>The Robin Hanson Chatbot 👇</h1>", unsafe_allow_html=True)

# initializing state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "system", "content": "You're analytical and kind. You want to help the user understand the economist Robin Hanson's worldview."}
    ]
if 'model_name' not in st.session_state:
    st.session_state['model_name'] = []
if 'cost' not in st.session_state:
    st.session_state['cost'] = []
if 'total_tokens' not in st.session_state:
    st.session_state['total_tokens'] = []
if 'total_cost' not in st.session_state:
    st.session_state['total_cost'] = 0.0

# sidebar for search - come back to it
st.sidebar.title("Suggestions")

# button to clear the chat history
clear_button = st.sidebar.button("Clear Conversation", key="clear")
    
# map model names to openai names later. gpt3.5 for now
model_name = "GPT-3.5"

if model_name == "GPT-3.5":
    model = "gpt-3.5-turbo"
else:
    model = "gpt-4"

# reset everything
if clear_button:
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['messages'] = [
        {"role": "system", "content": "You're analytical and kind. You want to help the user understand the economist Robin Hanson's worldview."}
    ]
    st.session_state['number_tokens'] = []
    st.session_state['model_name'] = []
    st.session_state['cost'] = []
    st.session_state['total_cost'] = 0.0
    st.session_state['total_tokens'] = []

def generate_response(query=None, messages=None):    
    
    posts, link, _ = get_hanson_articles(question=query, top_k=2)
    prompt = messages.copy()
    
    prompt.append({"role" : "user", "content" : f"I would like you to consider the following before answering the question: '{posts}'. Note that if the foregoing doesn't seem relevant, please inform the user that Hanson doesnt address that on his blog. Stick to what Hanson would say. {query}"})
    
    response = (
        openai
        .ChatCompletion
        .create(
            model="gpt-3.5-turbo",
            max_tokens=300,
            messages=prompt)
        )

    generated_text = response.choices[0].message.content
    
    st.session_state['messages'].append({"role" : "user", "content" : f'{query}'})
    st.session_state['messages'].append({"role" : "assistant", "content" : generated_text})
    
    total_tokens = response.usage.total_tokens
    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens
    
    return generated_text,  total_tokens, prompt_tokens, completion_tokens, link



# container for chat history
response_container = st.container()
# container for text box
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("You:", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        output, total_tokens, prompt_tokens, completion_tokens, links = generate_response(
            query=user_input, messages=st.session_state['messages'])
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)
        st.session_state['model_name'].append(model_name)
        st.session_state['total_tokens'].append(total_tokens)
        
        for i, link in enumerate(links):
            st.sidebar.write(f"[]({link})")

        if model_name == "GPT-3.5":
            cost = total_tokens * 0.002 / 1000
        else:
            cost = (prompt_tokens * 0.03 + completion_tokens * 0.06) / 1000

        st.session_state['cost'].append(cost)
        st.session_state['total_cost'] += cost

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            print(st.session_state['messages'])
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))
            st.write(
                f"Model used: {st.session_state['model_name'][i]}; Number of tokens: {st.session_state['total_tokens'][i]}; Cost: ${st.session_state['cost'][i]:.5f}")
            st.write("Links:")
            st.write(links[0] + "\n")
            st.write(links[1] + "\n")