# What Would Robin Say Chatbot

This repository contains a chatbot trained on Robin Hanson's blogging corpus and his Grabby Aliens paper. The chatbot uses vector embeddings to compute the cosine distance between the query and each blog post or chunk of the Grabby Aliens paper. It then merges the most relevant blog posts with the prompt template to focus ChatGPT when generating a reply.

The user's query (but not the blog posts -- too many tokens!) and the assistant's reply are added to the message history to keep context so the conversation can progress naturally. Links to the relevant blog posts are included below the chat history to allow the user to explore further.

The chat UI is created using Streamlit and Streamlit chat. Everything else is done using OpenAI, Pandas and Numpy.

# Requirements
OpenAI  
Pandas  
NumPy  
Streamlit  
Streamlit Chat

# Installation
Clone the repository  
Install the required packages using pip install -r requirements.txt

# Usage
Run streamlit run app.py  
Open the URL provided in the terminal in your web browser  
Start chatting with the bot

# Acknowledgements
Robin Hanson for providing the blogging corpus and the Grabby Aliens paper, OpenAI for providing the GPT-3.5 architecture used by the chatbot, NumPy for providing the vector computations, and Streamlit and Streamlit Chat for providing the chat UI

# License
This repository is licensed under the MIT License.
