#!/usr/bin/env python
from typing import Union, List, Tuple, Set
from spacy.tokens import Doc, Token, Span
from sense2vec.util import make_key, make_spacy_key
import spacy
from spacy.tokens import DocBin
from spacy.util import filter_spans
import plac
from wasabi import msg
from pathlib import Path
import tqdm

def merge_phrases(doc: Doc, cleaned_list) -> Doc:
    """Transform a spaCy Doc to match the sense2vec format: merge entities
    into one token and merge noun chunks without determiners.
    doc (Doc): The document to merge phrases in.
    RETURNS (Doc): The Doc with merged tokens.
    """

    spans = get_phrases(doc, cleaned_list)
    spans = filter_spans(spans)
    with doc.retokenize() as retokenizer:
        for span in spans:
            retokenizer.merge(span)
    return doc

def get_phrases(doc: Doc, cleaned_list) -> List[Span]:
    """Compile a list of sense2vec phrases based on a processed Doc: named
    entities and noun chunks without determiners.
    doc (Doc): The Doc to get phrases from.
    RETURNS (list): The phrases as a list of Span objects.
    """
    spans = list(doc.ents)
    ent_words: Set[str] = set()
    for span in spans:
        ent_words.update(token.i for token in span)
    # print('uy')
    for np in get_noun_phrases(doc, cleaned_list):
        # print('uy')
        # Prefer entities over noun chunks if there's overlap
        if not any(w.i in ent_words for w in np):
            spans.append(np)
    return spans

def get_noun_phrases(doc: Doc, cleaned_list) -> List[Span]:
    """Compile a list of noun phrases in sense2vec's format (without
    determiners). Separated out to make it easier to customize, e.g. for
    languages that don't implement a noun_chunks iterator out-of-the-box, or
    use different label schemes.
    doc (Doc): The Doc to get noun phrases from.
    RETURNS (list): The noun phrases as a list of Span objects.
    """
    spans = []
    if doc.is_parsed:
        for np in doc.noun_chunks:
            print(np.text.lower())
            if np.text.lower() in cleaned_list:
                print(np.text)
                spans.append(np)
    return spans

@plac.annotations(
    in_file=("Path to input file", "positional", None, str),
    out_dir=("Path to output directory", "positional", None, str),
    spacy_model=("Name of spaCy model to use", "positional", None, str),
    n_process=("Number of processes (multiprocessing)", "option", "n", int),
)
def main(in_file, out_dir, spacy_model="en_core_web_sm", n_process=1):
    """
    Step 2: Preprocess text in sense2vec's format

    Expects a binary .spacy input file consisting of the parsed Docs (DocBin)
    and outputs a text file with one sentence per line in the expected sense2vec
    format (merged noun phrases, concatenated phrases with underscores and
    added "senses").

    Example input:
    Rats, mould and broken furniture: the scandal of the UK's refugee housing

    Example output:
    Rats|NOUN ,|PUNCT mould|NOUN and|CCONJ broken_furniture|NOUN :|PUNCT
    the|DET scandal|NOUN of|ADP the|DET UK|GPE 's|PART refugee_housing|NOUN
    """
    cleaned_list = [line.rstrip() for line in open('np-resto-new', 'r')]
    input_path = Path(in_file)
    output_path = Path(out_dir)
    if not input_path.exists():
        msg.fail("Can't find input file", in_file, exits=1)
    if not output_path.exists():
        output_path.mkdir(parents=True)
        msg.good(f"Created output directory {out_dir}")
    nlp = spacy.load(spacy_model)
    msg.info(f"Using spaCy model {spacy_model}")
    with input_path.open("rb") as f:
        doc_bin_bytes = f.read()
    doc_bin = DocBin().from_bytes(doc_bin_bytes)
    msg.good(f"Loaded {len(doc_bin)} parsed docs")
    docs = doc_bin.get_docs(nlp.vocab)
    output_file = output_path / f"{input_path.stem}.s2v"
    lines_count = 0
    words_count = 0
    for doc in docs:
        print([token.text for token in doc])
    with output_file.open("w", encoding="utf8") as f:
        for doc in tqdm.tqdm(docs, desc="Docs", unit=""):
            doc = merge_phrases(doc, cleaned_list)
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
