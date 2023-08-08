# https://spacy.io/
# https://www.nltk.org/
# https://wordnet.princeton.edu/
# https://github.com/vacancy/SceneGraphParser

import re
from typing import Union
import spacy
from spacy.errors import Errors
from spacy.symbols import AUX, VERB, ADP, DET
from spacy.tokens import Doc, Span
from nltk.corpus import wordnet as wn
from scene_graph import SceneEntity, SceneGraph

def my_noun_chunks(doclike: Union[Doc, Span]):
    labels = [
        "oprd",
        "nsubj",
        "dobj",
        "nsubjpass",
        "pcomp",
        "pobj",
        "dative",
        "appos",
        "attr",
        "ROOT",
    ]
    doc = doclike.doc  # Ensure works on both Doc and Span.
    if not doc.has_annotation("DEP"):
        raise ValueError(Errors.E029)
    np_deps = [doc.vocab.strings.add(label) for label in labels]
    conj = doc.vocab.strings.add("conj")
    poss = doc.vocab.strings.add("poss")
    prev_end = -1
    for i, word in enumerate(doclike):
        if word.pos in (AUX, VERB, ADP, DET):
            continue
        # Prevent nested chunks from being produced
        if word.left_edge.i <= prev_end:
            continue
        if word.dep in np_deps:
            prev_end = word.i
            yield doc[word.left_edge.i : word.i + 1]
        elif word.dep == conj:
            head = word.head
            while head.dep == conj and head.head.i < head.i:
                head = head.head
            # If the head is an NP, and we're coordinated to it, we're an NP
            if head.dep in np_deps:
                prev_end = word.i
                yield doc[word.left_edge.i : word.i + 1]
        elif word.tag_ == 'NN' and word.dep == poss:
            yield doc[word.i : word.i + 1]

def prep_chunks(doclike: Union[Doc, Span]):
    phrasal_prep = {
        'in reference to', 
        'on top of', 
        'in regard to', 
        'on the side of', 
        'in front of', 
        'in addition to', 
        'on account of', 
        'with regard to', 
        'in spite of', 
        'on side of'
    }
    doc = doclike.doc  # Ensure works on both Doc and Span.
    if not doc.has_annotation("DEP"):
        raise ValueError(Errors.E029)
    for i, word in enumerate(doclike):
        if word.pos in (AUX, VERB, DET):
            continue
        second_superior = word.head.head
        # subj
        # prep: containing fake noun
        # obj: pobj  
        # E.g., A [woman] [in front of] a [piano].
        if (second_superior.dep_ == 'pobj' and 
            doc[second_superior.head.i:word.head.i + 1].text.lower() in phrasal_prep):                     
            yield doc[second_superior.head.i:word.head.i + 1]
        # verb
        # prt: (after verb)
        elif word.dep_ == 'prt' and word.head.pos_ == 'VERB' and word.head.i+1 == word.i:
            yield doc[word.head.i:word.i+1]

def merge_prep_chunks(doc: Doc) -> Doc:
    if not doc.has_annotation("DEP"):
        return doc
    with doc.retokenize() as retokenizer:
        for p in doc._.prep_chunks:
            attrs = {"tag": p.root.tag, "dep": p.root.dep}
            retokenizer.merge(p, attrs=attrs)  # type: ignore[arg-type]
    return doc

Doc.set_extension("my_noun_chunks", getter=my_noun_chunks, force=True)
Span.set_extension("my_noun_chunks", getter=my_noun_chunks, force=True)
Doc.set_extension("prep_chunks", getter=prep_chunks, force=True)
Span.set_extension("prep_chunks", getter=prep_chunks, force=True)

sceneSys = {"Synset('object.n.01')" : 0.7, "Synset('thing.n.12')" : 1, 
        "Synset('causal_agent.n.01')" : 1, "Synset('matter.n.03')" : 0.6,
        "Synset('attribute.n.02')" : 0.4, "Synset('group.n.01')" : 0.8,
        "Synset('measure.n.02')" : 0.2}

class SceneGraphParser():

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
            self.nlp.add_pipe("merge_entities")
        except OSError as e:
            raise ImportError('Unable to load the English model. \
                              Run `python -m spacy download en` first.') from e   

    def parse(self, sentence: str, sceneGraph: SceneGraph):
        def noun_descendant(token):
            for child in token.children:
                if child.dep_ == 'compound':
                    compounds.add(child.text)
                    noun_descendant(child)                                              
                elif child.dep_ == 'amod':
                    for c in self.__flatten_conjunction(child):
                        if wn.synsets(f"{c.text}_{word}"):
                            compounds.add(c.text)
                        else:
                            attributes.add(c.text)
                elif child.dep_ == 'det':
                    det.add(child.lemma_)
                elif child.dep_ == 'poss':
                    if child.pos_ == 'PRON':
                        try:
                            id = coreference[child.i]
                        except KeyError:
                            continue
                    else:
                        id = child.i
                    sub_entities.add((id, token.i))

        def get_sense_score(synsets, n):
            senses = 0
            count = 0
            for ss in synsets[:n]:
                if str(ss.pos()) == 'n':
                    count += 1
                    hypernym_paths = ss.hypernym_paths()
                    if hypernym_paths:
                        try: 
                            senses += sceneSys[str(hypernym_paths[0][2])]
                        except KeyError:
                            pass
            if count:
                return senses / count
            else:
                return 0

        def buffer_relation(subj, obj, relat):
            return {
                'subject': subj,
                'object': obj,
                'relation': relat,
            }

        def buffer_copula_attr(subj, obj):
            return {
                'subject': subj,
                'object': obj
            }

        bound = len(sentence)
        if bound > 30:
            bound = 30
        pattern = r"(photo|image|picture|painting) \S+ "
        match = re.search(pattern, sentence[:bound])
        if match:
            sentence = sentence[match.span()[1]:]        

        doc = merge_prep_chunks(self.nlp(sentence))

        # determine the subject entity of verbs
        verb_subjEnt = dict()
        subj_verbEnt = dict()
        coreference = dict()
        for token in doc:
            # subj: nsubj
            # verb: head of nsubj
            if token.dep_ == 'nsubj':
                verb_subjEnt[token.head.i] = token.i
                subj_verbEnt[token.i] = token.head.i
            # subj: head of acl    
            # verb: acl
            elif token.dep_ == 'acl':
                verb_subjEnt[token.i] = token.head.i
                subj_verbEnt[token.head.i] = token.i
            # verb: head of agent
            # subj: pobj
            elif token.dep_ == 'pobj' and token.head.dep_ == 'agent'\
                  and token.head.head.pos_ == 'VERB':
                verb_subjEnt[token.head.head.i] = token.i
                subj_verbEnt[token.i] = token.head.head.i
            # coreference in one sentence
            if token.pos_ == 'PRON':
                root = token.sent.root
                for prev_token in (root, *root.children):
                    if prev_token.i >= token.i:
                        break
                    if prev_token.pos_ == 'NOUN' and prev_token.morph.get('Number') == token.morph.get('Number')\
                        and (prev_token.dep_ not in ['dobj', 'pobj']):
                        coreference[token.i] = prev_token.i
                        break

        # find all the noun chunks as the entities
        noun_chunks = []
        sub_entities = set()
        potential_coreference = dict()
        for sent in doc.sents:
            for chunk in sent._.my_noun_chunks:
                token = chunk.root
                word = token.text
                sense_score = 0.5
                # compounds and attributes
                compounds = set()
                attributes = set()
                det = set()
                if token.pos_ == 'PRON':
                    if token.tag_ in ['DT', 'EX']:
                        continue
                elif token.pos_ == 'PROPN':
                    for child in token.children:                                         
                        if child.dep_ == 'amod':
                            for c in self.__flatten_conjunction(child):
                                attributes.add(c.text)
                else:
                    noun_descendant(token)

                    if compounds:
                        word = ' '.join(list(compounds)) + ' ' + word
                    else:
                        wordnet = wn.synsets(word)
                        if len(wordnet) > 0:    
                            sense_score = get_sense_score(wordnet, 3)
                            if not sense_score or word.lower() in \
                                ('photo', 'image', 'picture', 'painting'):
                                word = ''
                if word:
                    noun_chunks.append(token)
                    sceneEnt = SceneEntity()
                    sceneEnt.id = token.i + sceneGraph.id_count
                    sceneEnt.sense_score = sense_score
                    sceneEnt.entity = word.lower()
                    if det:
                        sceneEnt.det = det
                        if 'the' in sceneEnt.det:
                            for value in reversed(sceneGraph.entities.values()):
                                if value.entity == sceneEnt.entity and ('the' not in value.det):
                                    potential_coreference[sceneEnt.id] = value.id
                    if attributes:
                        sceneEnt.attributes |= attributes    
                    elif sceneEnt.sense_score > 0.5:
                        sceneGraph.add_to_wait(sceneEnt)
                    sceneGraph.entities[sceneEnt.id] = sceneEnt

        # determine all the relations among entities.
        relations = list()
        copula_attrs = list()
        for token in noun_chunks:
            relation = None
            copula_attr = None
            # subj: nsubj
            # verb: head of nsubj
            # obj: attr
            # E.g., The [woman] [is] a [pianist].
            if token.dep_ == 'attr' and token.head.i in verb_subjEnt:
                copula_attr = buffer_copula_attr(verb_subjEnt[token.head.i], token.i)
            # subj
            # verb
            # obj: dobj
            # E.g., A [woman] is [playing] the [piano].
            elif token.dep_ == 'dobj' and token.head.i in verb_subjEnt:
                relation = buffer_relation(verb_subjEnt[token.head.i], 
                                          token.i, 
                                          ((token.head.lemma_, token.head.pos_),))
            # obj: nsubjpass    
            # verb: head of agent
            # by
            # subj: pobj
            # E.g., The [piano] is [played] [by] a [woman].
            elif token.dep_ == 'nsubjpass' and token.head.i in verb_subjEnt:
                # Here, we reverse the passive phrase. I.e., subjpass -> obj and objpass -> subj.
                relation = buffer_relation(verb_subjEnt[token.head.i], 
                                           token.i, 
                                           ((token.head.lemma_, token.head.pos_),))
            # obj: pobj   
            elif token.dep_ == 'pobj':
                second_superior = token.head.head
                # (nsubjpass)
                if token.head.dep_ == 'agent':
                    pass                
                # subj: nsubj
                # verb: head of nsubj
                # prep: head of pobj, (after verb)
                # obj: pobj  
                elif (second_superior.pos_ == 'VERB'
                      and token.head.dep_ == 'prep' 
                      and second_superior.i + 1 == token.head.i
                      ) and second_superior.i in verb_subjEnt:
                    relation = buffer_relation(verb_subjEnt[second_superior.i], 
                                               token.i, 
                                               ((second_superior.lemma_, second_superior.pos_), 
                             (token.head.lemma_, token.head.pos_)))                                  
                # subj
                # verb
                # prep
                # obj: pobj
                # E.g., A [woman] playing the piano [in] the [room].
                elif (second_superior.pos_ == 'VERB' and second_superior.dep_ != 'ccomp'
                      and token.head.dep_ == 'prep') and second_superior.i in verb_subjEnt:
                    relation = buffer_relation(verb_subjEnt[second_superior.i], 
                                               token.i, 
                                               ((token.head.lemma_, token.head.pos_),))
                elif (second_superior.pos_ == 'VERB' and second_superior.dep_ == 'ccomp'
                      and token.head.dep_ == 'prep') and second_superior.i in verb_subjEnt:
                    for e in second_superior.rights:
                        if e.dep_ == 'dobj' and e in noun_chunks:
                            relation = buffer_relation(e.i, 
                                                    token.i, 
                                                    ((token.head.lemma_, token.head.pos_),))
                elif (second_superior.dep_ == 'acl' and second_superior.dep_ != 'ccomp'
                      and token.head.dep_ == 'prep') and second_superior.i in verb_subjEnt:
                    relation = buffer_relation(verb_subjEnt[second_superior.i], 
                                               token.i, 
                                               ((token.head.lemma_, token.head.pos_),))
                # subj
                # prep
                # obj: pobj
                # E.g., A [piano] in the [room].
                elif second_superior.pos_ == 'NOUN':
                    relation = buffer_relation(second_superior.i, 
                                               token.i, 
                                               ((token.head.lemma_, token.head.pos_),))
                # subj
                # a(dv)mod
                # obj: pobj
                # E.g., A [piano] [next to] a [woman].
                elif second_superior.dep_ in ('amod', 'advmod') and second_superior.head.pos_ == 'NOUN':
                    relation = buffer_relation(second_superior.head.i, 
                            token.i, 
                            ((second_superior.lemma_, second_superior.pos_), 
                             (token.head.lemma_, token.head.pos_)))
                # subj
                # verb
                # a(dv)mod
                # obj: pobj
                # E.g., A [woman] standing next to a [piano].
                elif (second_superior.dep_ in ('amod', 'advmod')
                      ) and second_superior.head.pos_ == 'VERB' and second_superior.head.i in verb_subjEnt:
                    relation = buffer_relation(verb_subjEnt[second_superior.head.i], 
                                               token.i, 
                                               ((second_superior.head.lemma_, second_superior.head.pos_),
                                                (second_superior.lemma_ + ' ' + token.head.lemma_, second_superior.pos_)))
                # subj
                # be
                # prep
                # obj: pobj
                # E.g., A [piano] is in the [room].
                elif second_superior.pos_ == 'AUX' and second_superior.i in verb_subjEnt:
                    relation = buffer_relation(verb_subjEnt[second_superior.i], 
                            token.i, 
                            ((token.head.lemma_, token.head.pos_),))

            if relation:
                relations.append(relation)
            
            if copula_attr:
                copula_attrs.append(copula_attr)

        # copula_attr
        for copula_attr in copula_attrs:
            flag = False
            try:
                subj = copula_attr['subject'] + sceneGraph.id_count
                obj = copula_attr['object'] + sceneGraph.id_count
                temp = sceneGraph.entities[obj].type
                sceneGraph.entities[obj].type = 0
                try:
                    sceneGraph.entities[subj].copula_attr.add(obj)
                    sceneGraph.entities[obj].copula_attr_parent.add(subj)
                    flag = True
                except KeyError:
                    sceneGraph.entities[obj].type = temp
            except KeyError:
                pass
            if flag:
                try:
                    sceneGraph.waiting_list.remove(obj)
                except ValueError:
                    pass
                try:
                    del subj_verbEnt[subj - sceneGraph.id_count]
                except KeyError:
                    pass
        
        imaginary_ent = set()

        # relation
        for relation in relations:
            subj = relation['subject']
            obj = relation['object']
            for x in self.__flatten_conjunction(doc[subj]):
                for y in self.__flatten_conjunction(doc[obj]):
                    subj = x.i + sceneGraph.id_count
                    obj = y.i + sceneGraph.id_count
                    if subj in sceneGraph.entities and obj in sceneGraph.entities:
                        relat = relation['relation']
                        if (('look', 'VERB') in relat 
                            and ('like', 'ADP') in relat) or (('resemble', 'VERB') in relat):
                            imaginary_ent.add(obj)
                        sceneGraph.relations[(subj, obj)].add(relat)
                        sceneGraph.entities[subj].subj_relations[(subj, obj)].add(relat)
                        sceneGraph.entities[obj].obj_relations[(subj, obj)].add(relat)
                        try:
                            del subj_verbEnt[subj - sceneGraph.id_count]
                        except KeyError:
                            pass

        # action
        for subj, verb in subj_verbEnt.items():
            subj += sceneGraph.id_count
            if subj in sceneGraph.entities:
                sceneGraph.entities[subj].actions.add(doc[verb].lemma_)
        
        # sub_entity
        for id1, id2 in list(sub_entities):
            id1 += sceneGraph.id_count
            id2 += sceneGraph.id_count
            try:
                temp = sceneGraph.entities[id2].type
                sceneGraph.entities[id2].type = 0
                try:
                    sceneGraph.waiting_list.remove(id2)
                except ValueError:
                    pass
                try:
                    sceneGraph.entities[id1].sub_entity.add(id2)
                    sceneGraph.entities[id2].sub_entity_parent.add(id1)
                except KeyError:
                    sceneGraph.entities[id2].type = temp
            except KeyError:
                pass
        
        # coreference
        for id2, id1 in coreference.items():
            id1 += sceneGraph.id_count
            id2 += sceneGraph.id_count
            sceneGraph.merge(id1, id2)
            try:
                sceneGraph.waiting_list.remove(id2)
            except ValueError:
                pass

        for id2, id1 in potential_coreference.items():
            sceneGraph.merge(id1, id2)
            try:
                sceneGraph.waiting_list.remove(id2)
            except ValueError:
                pass

        # imaginary entity
        for e in list(imaginary_ent):
            sceneGraph.entities[e].type = -1
            try:
                sceneGraph.waiting_list.remove(e)
            except ValueError:
                pass
            for subj, obj in sceneGraph.entities[e].subj_relations.keys():
                sceneGraph.entities[obj].type = -1
                try:
                    sceneGraph.waiting_list.remove(obj)
                except ValueError:
                    pass

        sceneGraph.id_count += doc[-1].i

        return sceneGraph

    @staticmethod
    def __flatten_conjunction(node):
        yield node
        for c in node.children:
            if c.dep_ == 'conj':
                yield c