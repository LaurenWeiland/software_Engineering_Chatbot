import spacy

nlp = spacy.load("en_core_web_sm")

# Process whole documents
'''
text = ("When Sebastian Thrun started working on self-driving cars at "
        "Google in 2007, few people outside of the company took him "
        "seriously. “I can tell you very senior CEOs of major American "
        "car companies would shake my hand and turn away because I wasn’t "
        "worth talking to,” said Thrun, in an interview with Recode earlier "
        "Hello my name is Lauren")
'''
text = "I want to order some Greek Pizza"
doc = nlp(text)
textcat = nlp.add_pipe("textcat")

# Analyze syntax
'''
print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])
'''
# Find named entities, phrases and concepts

for entity in doc:
    print(entity.text, entity.pos_,entity.dep_)

for token in doc:
    if token.dep_ == "dobj":
        print(token.head.text + " " + token.text)
    
