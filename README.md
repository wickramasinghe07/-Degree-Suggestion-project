# Degree-Suggestion-project
Final year research


Natural Language Processing

    FrontEnd REACT Framework Next.js
    BackEnd  Python

FrontEnd 

        npm i node-modules
        npm run dev

BackEnd

         pip install -r requirements.txt
         cd backend
         uvicorn main:app
         numpy==1.26.4
         opencv-python==4.8.0.76
         pip install mediapipe==0.10.3
         pip install openai
         pip install spacy
         python -m spacy download en_core_web_sm
        

Natural language processing (NLP) is a field that focuses on making natural human language usable by computer programs. NLTK, or Natural Language Toolkit, is a Python package that you can use for NLP

    Find text to analyze
    Preprocess my text for analysis
    Analyze your text
    Create visualizations based on my analysis

Installing version 3.5

     $ python -m pip install nltk==3.5

Tokenizing

Tokenizing by word: Words are like the atoms of natural language. They’re the smallest unit of meaning that still makes sense on its own. Tokenizing your text by word allows you to identify words that come up particularly often. For example, if you were analyzing a group of job ads, then you might find that the word “Python” comes up often. That could suggest high demand for Python knowledge, but you’d need to look deeper to know more

Tokenizing by sentence: When you tokenize by sentence, you can analyze how those words relate to one another and see more context.

     from nltk.tokenize import sent_tokenize, word_tokenize

Filtering Stop Words
   
     >>> nltk.download("stopwords")
     >>> from nltk.corpus import stopwords
     >>> from nltk.tokenize import word_tokenize

  quote from Worf 

    >>> worf_quote = "Sir, I protest. I am not a merry man!"

Stemming

Stemming is a text processing task in which you reduce words to their root, which is the core part of a word. 

     from nltk.stem import PorterStemmer
     from nltk.tokenize import word_tokenize
     
Lemmatizing

ircle back to lemmatizing. Like stemming, lemmatizing reduces words to their core meaning, but it will give you a complete English word 

    >>> from nltk.stem import WordNetLemmatizer
    >>> lemmatizer = WordNetLemmatizer()
    >>> lemmatizer.lemmatize("scarves")
        'scarf'


    
    
    
