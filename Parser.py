import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))
model = load_model('chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r> ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse = True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])[1]})

    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result
#nlp = spacy.load("en_core_web_sm")

'''
Parse takes the user input from the website/chatbot and starts
classifying the data passed in. This will use the _Classify and
_PassClassify methods to make the library send the desired
information to the Chatbot
'''
def Parse():
    pass


'''
Classify does the heavy lifting of this object. It takes the
parsed input from the website/chatbot and breaks it into a
meaningful set of information to have the library of resources
process and send to the chatbot.
'''


def _Classify():
    pass


'''
Pass Classify takes the classified response and sends it to
the Resource Library. This should be tagged info used by the
library to send to the chatbot
'''
def _PassClassify():
    pass



if __name__ == '__main__':
    while True:
        message = input('')
        ints = predict_class(message)
        res = get_response(ints,intents)
        print(res)
