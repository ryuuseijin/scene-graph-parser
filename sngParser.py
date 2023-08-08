# https://github.com/vacancy/SceneGraphParser

import os
import spacy
from copy import deepcopy

cwd = os.path.dirname(__file__)

_caches = dict()

verb_path = 'phrasal-verbs.txt'
verb_path = os.path.abspath(os.path.join(cwd, verb_path))

prep_path = 'phrasal-preps.txt'
prep_path = os.path.abspath(os.path.join(cwd, prep_path))

noun_path = 'scene-nouns.txt'
noun_path = os.path.abspath(os.path.join(cwd, noun_path))

def load_list(filename):
    if filename not in _caches:
        out = set()
        for x in open(filename):
            x = x.strip()
            if len(x) > 0:
                out.add(x)
        _caches[filename] = out
    return _caches[filename]

def is_phrasal_verb(verb):
    return verb in load_list(verb_path)

def is_phrasal_prep(prep):
    return prep in load_list(prep_path)

def is_scene_noun(noun):
    head = noun.split(' ')[-1]
    s = load_list(noun_path)
    return noun in s or head in s

class SpacyParser():

    def __init__(self, model=None):
        if spacy.__version__ < '3':
            default_model = 'en'
        else:
            default_model = 'en_core_web_sm'

        self.model = model
        if self.model is None:
            self.model = default_model

        try:
            self.nlp = spacy.load(self.model)
        except OSError as e:
            raise ImportError('Unable to load the English model. Run `python -m spacy download en` first.') from e
        
    

    def parse(self, sentence, return_doc=False):

        doc = self.nlp(sentence)

        # Step 1: find all the noun chunks as the entities, and resolve the modifiers on them.
        entities = list()
        entity_chunks = list()
        for entity in doc.noun_chunks:
            # Ignore pronouns such as "it".
            if entity.root.lemma_ == '-PRON-':
                continue

            ent = dict(
                span=entity.text,
                lemma_span=entity.lemma_,
                head=entity.root.text,
                lemma_head=entity.root.lemma_,
                span_bounds=(entity.start, entity.end),
                modifiers=[]
            )

            def dfs(node):
                for x in node.children:
                    if x.dep_ == 'det':
                        ent['modifiers'].append({'dep': x.dep_, 'span': x.text, 'lemma_span': x.lemma_})
                    elif x.dep_ == 'nummod':
                        ent['modifiers'].append({'dep': x.dep_, 'span': x.text, 'lemma_span': x.lemma_})
                    elif x.dep_ == 'amod':
                        for y in self.__flatten_conjunction(x):
                            ent['modifiers'].append({'dep': x.dep_, 'span': y.text, 'lemma_span': y.lemma_})
                    elif x.dep_ == 'compound':
                        ent['head'] = x.text + ' ' + ent['head']
                        ent['lemma_head'] = x.lemma_ + ' ' + ent['lemma_head']
                        dfs(x)

            dfs(entity.root)

            if is_scene_noun(ent['lemma_head']):
                ent['type'] = 'scene'
            else:
                ent['type'] = 'unknown'

            entities.append(ent)
            entity_chunks.append(entity)

        # Step 2: determine the subject of verbs (including nsubj, acl and pobjpass).
        # To handle the situation where multiple nouns may be the same word,
        # the tokens are represented by their position in the sentence instead of their text.
        relation_subj = dict()
        for token in doc:
            # E.g., A [woman] is [playing] the piano.
            if token.dep_ == 'nsubj':
                relation_subj[token.head.i] = token.i
            # E.g., A [woman] [playing] the piano...
            elif token.dep_ == 'acl':
                relation_subj[token.i] = token.head.i
            # E.g., The piano is [played] by a [woman].
            elif token.dep_ == 'pobj' and token.head.dep_ == 'agent' and token.head.head.pos_ == 'VERB':
                relation_subj[token.head.head.i] = token.i

        # Step 3: determine all the relations among entities.
        relations = list()
        fake_noun_marks = set()
        for entity in doc.noun_chunks:
            # Again, the subjects and the objects are represented by their position.
            relation = None

            # E.g., A woman is [playing] the [piano].
            # E.g., The woman [is] a [pianist].
            if entity.root.dep_ in ('dobj', 'attr') and entity.root.head.i in relation_subj:
                relation = {
                    'subject': relation_subj[entity.root.head.i],
                    'object': entity.root.i,
                    'relation': entity.root.head.text,
                    'lemma_relation': entity.root.head.lemma_
                }
            elif entity.root.dep_ == 'pobj':
                # E.g., The piano is played [by] a [woman].
                if entity.root.head.dep_ == 'agent':
                    pass
                # E.g., A [woman] is playing with the piano in the [room].
                elif (
                        entity.root.head.head.pos_ == 'VERB' and
                        entity.root.head.head.i + 1 == entity.root.head.i and
                        is_phrasal_verb(entity.root.head.head.lemma_ + ' ' + entity.root.head.lemma_)
                ) and entity.root.head.head.i in relation_subj:
                    relation = {
                        'subject': relation_subj[entity.root.head.head.i],
                        'object': entity.root.i,
                        'relation': entity.root.head.head.text + ' ' + entity.root.head.text,
                        'lemma_relation': entity.root.head.head.lemma_ + ' ' + entity.root.head.lemma_
                    }
                # E.g., A [woman] is playing the piano in the [room]. Note that room.head.head == playing.
                # E.g., A [woman] playing the piano in the [room].
                elif (
                        entity.root.head.head.pos_ == 'VERB' or
                        entity.root.head.head.dep_ == 'acl'
                ) and entity.root.head.head.i in relation_subj:
                    relation = {
                        'subject': relation_subj[entity.root.head.head.i],
                        'object': entity.root.i,
                        'relation': entity.root.head.text,
                        'lemma_relation': entity.root.head.lemma_
                    }
                # E.g., A [woman] in front of a [piano].
                elif (
                        entity.root.head.head.dep_ == 'pobj' and
                        is_phrasal_prep(doc[entity.root.head.head.head.i:entity.root.head.i + 1].text.lower())
                ):
                    fake_noun_marks.add(entity.root.head.head.i)
                    relation = {
                        'subject': entity.root.head.head.head.head.i,
                        'object': entity.root.i,
                        'relation': doc[entity.root.head.head.head.i:entity.root.head.i + 1].text,
                        'lemma_relation': doc[entity.root.head.head.head.i:entity.root.head.i].lemma_
                    }
                # E.g., A [piano] in the [room].
                elif entity.root.head.head.pos_ == 'NOUN':
                    relation = {
                        'subject': entity.root.head.head.i,
                        'object': entity.root.i,
                        'relation': entity.root.head.text,
                        'lemma_relation': entity.root.head.lemma_
                    }
                # E.g., A [piano] next to a [woman].
                elif entity.root.head.head.dep_ in ('amod', 'advmod') and entity.root.head.head.head.pos_ == 'NOUN':
                    relation = {
                        'subject': entity.root.head.head.head.i,
                        'object': entity.root.i,
                        'relation': entity.root.head.head.text + ' ' + entity.root.head.text,
                        'lemma_relation': entity.root.head.head.lemma_ + ' ' + entity.root.head.lemma_
                    }
                # E.g., A [woman] standing next to a [piano].
                elif entity.root.head.head.dep_ in ('amod', 'advmod') and entity.root.head.head.head.pos_ == 'VERB' and entity.root.head.head.head.i in relation_subj:
                    relation = {
                        'subject': relation_subj[entity.root.head.head.head.i],
                        'object': entity.root.i,
                        'relation': entity.root.head.head.text + ' ' + entity.root.head.text,
                        'lemma_relation': entity.root.head.head.lemma_ + ' ' + entity.root.head.lemma_
                    }
                # E.g., A [woman] is playing the [piano] in the room
                elif entity.root.head.head.dep_== 'VERB' and entity.root.head.head.i in relation_subj:
                    relation = {
                        'subject': relation_subj[entity.root.head.head.i],
                        'object': entity.root.i,
                        'relation': entity.root.head.text,
                        'lemma_relation': entity.root.head.lemma_
                    }
                # E.g., A [piano] is in the [room].
                elif entity.root.head.head.pos_ == 'AUX' and entity.root.head.head.i in relation_subj:
                    relation = {
                        'subject': relation_subj[entity.root.head.head.i],
                        'object': entity.root.i,
                        'relation': entity.root.head.text,
                        'lemma_relation': entity.root.head.lemma_
                    }

            # E.g., The [piano] is played by a [woman].
            elif entity.root.dep_ == 'nsubjpass' and entity.root.head.i in relation_subj:
                # Here, we reverse the passive phrase. I.e., subjpass -> obj and objpass -> subj.
                relation = {
                    'subject': relation_subj[entity.root.head.i],
                    'object': entity.root.i,
                    'relation': entity.root.head.text,
                    'lemma_relation': entity.root.head.lemma_
                }

            if relation is not None:
                relations.append(relation)

        # Apply the `fake_noun_marks`.
        entities = [e for e, ec in zip(entities, entity_chunks) if ec.root.i not in fake_noun_marks]
        entity_chunks = [ec for ec in entity_chunks if ec.root.i not in fake_noun_marks]

        filtered_relations = list()
        for relation in relations:
            # Use a helper function to map the subj/obj represented by the position
            # back to one of the entity nodes.
            for x in self.__flatten_conjunction(doc[relation['subject']]):
                for y in self.__flatten_conjunction(doc[relation['object']]):
                    rel = deepcopy(relation)
                    rel['subject'] = self.__locate_noun(entity_chunks, x.i)
                    rel['object'] = self.__locate_noun(entity_chunks, y.i)
                    if rel['subject'] != None and rel['object'] != None:
                        filtered_relations.append(rel)

        if return_doc:
            return {'entities': entities, 'relations': filtered_relations}, doc
        return {'entities': entities, 'relations': filtered_relations}

    @staticmethod
    def __locate_noun(chunks, i):
        for j, c in enumerate(chunks):
            if c.start <= i < c.end:
                return j
        return None

    @staticmethod
    def __flatten_conjunction(node):
        yield node
        for c in node.children:
            if c.dep_ == 'conj':
                yield c