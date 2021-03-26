import nltk
import numpy as np
import random
import string

#i pip install nltk and numpy

f = open('chatbot.txt', 'r', errors = 'ignore')
raw = f.read()

raw = raw.lower()

nltk.download('punkt')
nltk.download('wordnet')

sent_tokens = nltk.sent_tokenize(raw)
word_tokens = nltk.word_tokenize(raw)

#print(word_tokens[:2])


#preprocessing and handling raw text
lemmer = nltk.stem.WordNetLemmatizer()
def lemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def lemNormalize(text):
        return lemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


#programming a greet response
Greeting_inputs = ('hello', 'hi', 'greeting', 'sup', 'whasup', 'whassup',"what's up", 'hey')
Greeting_responses = ['hi', 'hey', 'hi there', 'hello', 'how you dey', 'how fa na']

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in Greeting_inputs:
            return random.choice(Greeting_responses)

#generate response
#i had to pip install sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity

#writing the response

def response(user_response):
    robo_response = ' '
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer = lemNormalize, stop_words = 'english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if ( req_tfidf == 0 ):
        robo_response = robo_response + "I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response + sent_tokens[idx]
        return robo_response

#programming start and end point of conversation
flag = True
print("Chatbot: My name is Chatbot! I can help you with Chatbots. If you want to quit, just type Bye anytime")
while(flag ==True):
    user_response = input()
    user_response = user_response.lower()
    if (user_response != 'bye'):
        if (user_response == 'thanks' or user_response == 'thank you'):
            flag = False
            print("Chatbot: You are welcome")
        else:
            if (greeting(user_response) != None):
                print("Chatbot: " + greeting(user_response))
            else:
                print("SampleBot: ", end = " ")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag = False
        print("SampleBot: Thanks for talking bye...")


