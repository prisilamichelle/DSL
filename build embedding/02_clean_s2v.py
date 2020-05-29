import re
from flashtext import KeywordProcessor
from nltk.corpus import stopwords

def clean_token(text):
    stop_words = stopwords.words('english')
    sentences = []
    for sentence in text.split('\n'):
        words = []
        for word in sentence.split():
            l = word.split('|')
            if l[0] not in stop_words and l[1] not in ['AUX', 'DET', 'ADP', 'PART', 'SCONJ', 'CCONJ', 'PRON', 'GPE', 'DATE', 'TIME', 'NUM', 'CARDINAL', 'ORDINAL', 'QUANTITY', 'PUNCT', 'X', 'SYM', 'PERSON', 'MONEY', 'PERCENT', 'FAC', 'LOC']:
                has_number = re.search('\d', l[0])
                if has_number == None or '_' in l[0]:
                    l[0] = l[0].lower()
                    words.append('|'.join(l))
                else:
                    if re.search('\d', l[0]).span()[0] != 0:
                        l[0] = l[0].lower()
                        words.append('|'.join(l))
        sentences.append(' '.join(words))
    sentences = list(filter(None, sentences))
    text = '\n'.join(sentences)
    return text

def replace_contractions(text):
    text = text.replace("'ll|VERB", "will|VERB")
    text = text.replace("'d|VERB", "would|VERB")
    text = text.replace("wo|VERB", "will|VERB")
    text = text.replace("ca|VERB", "can|VERB")
    return text

filename = input('Insert filename : ')
# read s2v
with open(filename, 'r') as in_file:
    text = in_file.read()
# replace contractions
text = replace_contractions(text)
# clean all token
text = clean_token(text) 

# output fixed s2v
with open(filename, 'w') as out_file:
    out_file.write(text)