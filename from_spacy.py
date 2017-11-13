import spacy
from nltk.corpus import stopwords
nlp = spacy.load('en')
from bs4 import BeautifulSoup as bs

def sent_tokenize(text):
    text = nlp(text)
    sent_list = [sent for sent in text.sents]
    return sent_list


def entity_search(text):
    sent_list = []
    for sent in sent_tokenize(text):
        for word in sent:
            if word.ent_type_:
                s = (word.orth_,word.ent_type_)
            else:
                s = word.orth_
            sent_list.append(s)

    return sent_list




def character_searching(text):
    text = bs(text).get_text()
    c = nlp(text)
    person_list = [i.orth_ for i in c.ents if i.label_ == "PERSON"]
    char_list = []
    for i in c:
        if i.dep_ == 'nsubj':
            li = [' '.join([t.orth_ for t in i.lefts]+[i.orth_]),
                  i.head.orth_]
            char_list.append(li)
        if i.dep_ == 'compound' and str(i.orth_+' ' +i.head.orth_) in person_list:
            li = [str(i.orth_+' ' +i.head.orth_) ,' ']
            char_list.append(li)

    print(char_list)
    return char_list



#-----------------------------------------------------------------------------------------

example2 = "Apple's Company stocks dropped dramatically after the death of Steve Jobs in 7th October 1994."
example = "The boy with the spotted dog quickly ran after the firetruck."
ex3 = "The boy with"
print(character_searching(ex3))