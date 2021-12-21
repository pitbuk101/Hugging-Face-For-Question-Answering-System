# Question Answering System



Problem Description

Today the world is full of articles on large variety of topics. We aim to build a question- answering product that can that can understand the information in these articles and answer some simple questions related to those articles.

Proposed Solution

We plan to use Natural Language Processing techniques to extract the sematic & syntactic information from these articles and use them to find closest answer to the user’s question. We’ll extract NLP features like POS tags, lemmas, synonyms, hypernyms, meronyms, etc. for every sentence, and use Solr tool to store & index all this information. We’ll extract the same features from the question and form a Solr search query. This query will fetch the answer from the indexed Solr objects.



The logic behind training two models - the former is a conditional model, trained only on correct question/answers pairs, 
while the latter additionally includes tricky questions with answers that can't be found in the context. 
The idea is that combining the output of both models will improve the discrimination ability on impossible questions.

## Web application 

Explore the QA system using application hosted on Streamlit Sharing:
https://share.streamlit.io/snexus/nlp-question-answering-system/main

