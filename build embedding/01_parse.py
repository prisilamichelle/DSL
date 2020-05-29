#!/usr/bin/env python
import tqdm
import plac
import spacy
import textacy
import regex as re
import webcolors
from wasabi import msg
from pathlib import Path
from typing import Union, List, Tuple, Set
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from spacy.util import filter_spans
from spacy.tokens import Doc, Token, Span
from sense2vec.util import make_key, get_true_cased_text

def common_member(a, b): 
    a_set = set(a) 
    b_set = set(b) 
    if (a_set & b_set): 
        return True 
    else: 
        return False

def get_adjective_phrases(doc):
  patterns = [{"POS": "ADV"}, {"POS": "ADV", "OP": "?"}, {"POS": "ADJ"}, {"POS": "ADV", "OP": "?"}]
  matches = textacy.extract.matches(doc, patterns)
  adj = []
  for phrase in matches:
    tag = [token.tag_ for token in phrase]
    pos = [token.pos_ for token in phrase]
    has_number = re.search('\d', phrase.text)
    if not common_member(tag, ['WRB']) and has_number == None:
        adj.append(phrase)
  # print('ADJ', adj)
  return adj

def get_noun_phrases(doc, wn_lemmas, special_ents):
  noun = []
  print('NOUN CHUNKS : ', list(doc.noun_chunks))
  for np in doc.noun_chunks:
    if len(np) >= 2:
      pos = [token.pos_ for token in np]
      dep = [token.dep_ for token in np]
      has_punct = re.search('[,.]', np.text)
      has_number = re.search('\d', np.text)

      if has_punct == None and not common_member(pos, ['SPACE', 'INTJ', 'SYM']):
        if pos[0] not in ['AUX', 'ADP', 'PRON']:
          # print(np.text)
          i = 0
          pos_element = pos[i]
          if 'X' in pos:
            print('X warning ' + np.text + str([token.pos_ for token in np]) + str([token.dep_ for token in np]))

          while pos_element in ['X', 'CCONJ', 'SCONJ', 'DET', 'PUNCT']:
            if pos_element == 'X':
              if dep[i] == 'nummod':
                np = np[1:]
            else:
              np = np[1:]
            i+=1
            pos_element = pos[i]
            
          # print(np.text)
          # print(np.text + str([token.pos_ for token in np]) + str([token.dep_ for token in np]))
          if np[0].pos_ != 'NUM' and len(np) >= 2:
            if np[0].dep_ not in ['advmod', 'amod']:
              noun.append(np)
            elif np[0].dep_ == 'amod' and np[0].pos_ not in ['ADJ', 'ADV']:
              noun.append(np)
            elif '_'.join(np.text.split()) in wn_lemmas:
              noun.append(np)
            elif np[0].i in special_ents:
              noun.append(np)
            elif np[0].text in webcolors.CSS3_NAMES_TO_HEX:
              noun.append(np)
              # print('COLOR ' + np.text + str([token.pos_ for token in np]) + str([token.tag_ for token in np]) + str([token.dep_ for token in np]))
            elif np[0].pos_ == 'ADJ' and wordnet.synsets(np[0].text):
              # disallowed_lexnames = ['noun.person', 'noun.act', 'noun.attribute', 'noun.quantity', 'noun.plant', 'noun.body', 'noun.cognition', 'verb.change', 'verb.competition', 'verb.cognition', 'verb.emotion', 'verb.feeling']
              allowed_lexnames = ['noun.artifact', 'noun.shape', 'noun.relation', 'noun.event', 'noun.phenomenon', 'noun.communication', 'verb.contact', 'verb.stative', 'verb.consumption']
              synsets = [synset for synset in wordnet.synsets(np[0].text) if synset.name().split('.')[0] == np[0].text and synset.lexname() in allowed_lexnames]
              if not synsets:
                synsets = [synset for synset in wordnet.synsets(np[0].text) if synset.name().split('.')[0] in np[0].text and synset.lexname() in allowed_lexnames]
              pos_synsets = [synset.pos() for synset in synsets]
              lexname_synsets = [synset.name().split('.')[0] + ' ' + synset.lexname() for synset in synsets]
              if common_member(pos_synsets, ['v', 'n']):
                noun.append(np)
                print(lexname_synsets)
                print('BY SYNSETS ' + np.text + str([token.pos_ for token in np]) + str([token.tag_ for token in np]) + str([token.dep_ for token in np]))
            # elif '_' in np[0].text:
              # noun.append(np)
              # print('BY PHRASE ' + np.text + str([token.pos_ for token in np]) + str([token.tag_ for token in np]) + str([token.dep_ for token in np]))
            else:
              if np[0].dep_ == 'amod':
                print('AMOD ' + np.text + str([token.pos_ for token in np]) + str([token.tag_ for token in np]) + str([token.dep_ for token in np]))
              
              wnl = WordNetLemmatizer()
              joined_phrase = '_'.join(np.text.split())
              if wnl.lemmatize(joined_phrase) != joined_phrase:
                noun.append(np)
              else:
                if len(np) > 2:
                  wordnet_phrase = False
                  norp_phrase = False
                  pos_element = np[0].pos_
                  while pos_element in ['ADV', 'ADJ', 'CCONJ', 'DET', 'PUNCT'] and len(np) > 2:
                    np = np[1:]
                    if np[0].i in special_ents:
                      norp_phrase = True
                      break
                    joined_phrase = '_'.join(np.text.split())
                    if wnl.lemmatize(joined_phrase) != joined_phrase or joined_phrase in wn_lemmas:
                      wordnet_phrase = True
                      break
                    pos_element = np[0].pos_
                  if wordnet_phrase or norp_phrase:
                    noun.append(np)
                    # print('JOINED TOO ' + np.text + str([token.pos_ for token in np]) + str([token.tag_ for token in np]) + str([token.dep_ for token in np]))
                  else:
                    if np[0].pos_ in ['PROPN']:
                      noun.append(np)
                      print('PROPN ' + np.text + str([token.pos_ for token in np]) + str([token.tag_ for token in np]) + str([token.dep_ for token in np]))
                    # else:
                        # print('ADJ REMOVED ' + np.text + str([token.pos_ for token in np]) + str([token.tag_ for token in np]) + str([token.dep_ for token in np]))
                # else:
                  # print('NOT IN WORDNET ' + np.text + str([token.pos_ for token in np]) + str([token.tag_ for token in np]) + str([token.dep_ for token in np]))
          else:
            if has_number != None:
              print('NUMBER ALERT ' + np.text + str([token.pos_ for token in np]) + str([token.tag_ for token in np]) + str([token.dep_ for token in np]))
        else:
          print('WEIRD PHRASE ' + np.text + str([token.pos_ for token in np]) + str([token.tag_ for token in np]) + str([token.dep_ for token in np]))
  # print('NOUN : ', str(noun))
  return noun

def merge_phrases(doc: Doc, spans) -> Doc:
    """Transform a spaCy Doc to match the sense2vec format: merge entities
    into one token and merge noun chunks without determiners.
    doc (Doc): The document to merge phrases in.
    RETURNS (Doc): The Doc with merged tokens.
    """
    with doc.retokenize() as retokenizer:
        for span in spans:
            retokenizer.merge(span)
    return doc

def get_phrases(doc: Doc, wn_lemmas) -> List[Span]:
    """Compile a list of sense2vec phrases based on a processed Doc: named
    entities and noun chunks without determiners.
    doc (Doc): The Doc to get phrases from.
    RETURNS (list): The phrases as a list of Span objects.
    """
    entities = list(doc.ents).copy()
    
    # a = len(entities)
    special_ents: Set[str] = set()
    for ent in list(doc.ents):
        if ent.label_ in ['NORP', 'ORDINAL']:
            special_ents.update(token.i for token in ent)
        elif ent.label_ in ['WORK_OF_ART', 'LAW', 'EVENT', 'LANGUAGE', 'PRODUCT', 'ORG', 'NORP', 'FAC', 'LOC']:
            # print('ENT REMOVED', ent.text, ent.label_)
            entities.remove(ent)
    print('ENT WORDS ', entities)
    # print('NORP : ', norp)
    # b = len(entities)
    # if a != b:
    #   print('ENT WORDS 2', entities)  
    spans = get_noun_phrases(doc, wn_lemmas, special_ents)
    np_words: Set[str] = set()
    for span in spans:
        np_words.update(token.i for token in span)
    for ent in entities:
        # Prefer noun chunks over entities if there's overlap
        if not any(w.i in np_words for w in ent):
            spans.append(ent)
    return spans

def make_spacy_key(
    obj: Union[Token, Span], prefer_ents: bool = False, lemmatize: bool = False
) -> Tuple[str, str]:
    """Create a key from a spaCy object, i.e. a Token or Span. If the object
    is a token, the part-of-speech tag (Token.pos_) is used for the sense
    and a special string is created for URLs. If the object is a Span and
    has a label (i.e. is an entity span), the label is used. Otherwise, the
    span's root part-of-speech tag becomes the sense.
    obj (Token / Span): The spaCy object to create the key for.
    prefer_ents (bool): Prefer entity types for single tokens (i.e.
        token.ent_type instead of tokens.pos_). Should be enabled if phrases
        are merged into single tokens, because otherwise the entity sense would
        never be used.
    lemmatize (bool): Use the object's lemma instead of its text.
    RETURNS (unicode): The key.
    """
    default_sense = "?"
    text = get_true_cased_text(obj, lemmatize=lemmatize)
    if isinstance(obj, Token):
        if obj.like_url:
            text = "%%URL"
            sense = "X"
        elif obj.ent_type_ and obj.ent_type_ not in ['WORK_OF_ART', 'LAW', 'EVENT', 'LANGUAGE', 'PRODUCT', 'ORG', 'NORP', 'FAC', 'LOC'] and prefer_ents:
            sense = obj.ent_type_
        else:
            sense = obj.pos_
    elif isinstance(obj, Span):
        sense = obj.label_ or obj.root.pos_
    return (text, sense or default_sense)

@plac.annotations(
    in_file=("Path to input file", "positional", None, str),
    out_dir=("Path to output directory", "positional", None, str),
    spacy_model=("Name of spaCy model to use", "positional", None, str),
    n_process=("Number of processes (multiprocessing)", "option", "n", int),
)
def main(in_file, out_dir, spacy_model="en_core_web_sm", n_process=1):
    """
    Step 1: Parse raw text with spaCy

    Expects an input file with one sentence per line and will output a .spacy
    file of the parsed collection of Doc objects (DocBin).
    """
    input_path = Path(in_file)
    output_path = Path(out_dir)
    if not input_path.exists():
        msg.fail("Can't find input file", in_file, exits=1)
    if not output_path.exists():
        output_path.mkdir(parents=True)
        msg.good(f"Created output directory {out_dir}")
    nlp = spacy.load(spacy_model)
    msg.info(f"Using spaCy model {spacy_model}")
    msg.text("Preprocessing text...")
    texts = [line.rstrip() for line in open(in_file, 'r')] 
    docs = nlp.pipe(texts, n_process=n_process)
    output_file = output_path / f"{input_path.stem}.s2v"
    lines_count = 0
    words_count = 0
    wn_lemmas = set(wordnet.all_lemma_names())
    with output_file.open("w", encoding="utf8") as f:
        for doc in tqdm.tqdm(docs, desc="Docs", unit=""):
            print(doc)
            spans = get_phrases(doc, wn_lemmas)
            spans = filter_spans(spans)
            print('NOUN SPAN', str(spans))
            doc = merge_phrases(doc, spans)
            spans = get_adjective_phrases(doc)
            spans = filter_spans(spans)
            print('ADJ SPAN', str(spans))
            print('*-----------------------------------------*')
            doc = merge_phrases(doc, spans)
            words = []
            for token in doc:
                if not token.is_space:
                    word, sense = make_spacy_key(token, prefer_ents=True)
                    words.append(make_key(word, sense))
            f.write(" ".join(words) + "\n")
            lines_count += 1
            words_count += len(words)
    msg.good(
        f"Successfully preprocessed {lines_count} docs ({words_count} words)",
        output_file.resolve(),
    )

if __name__ == "__main__":
    plac.call(main)
