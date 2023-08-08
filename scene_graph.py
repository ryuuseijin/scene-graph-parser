from collections import defaultdict
from pyvis.network import Network

# real entity: LightPink '#ffaeb9'
# imaginary entity: Orange '#ffa500'
# copula_attr: smaller star
# sub_entity: smaller dot
# attribute: LightGreen '#a8dda8' smaller dot
#
# relation: LightBlue '#bfefff'
# action: LightBlue '#bfefff' diamond

class SceneEntity():
    def __init__(self):
        self.id: int
        self.sense_score: float
        self.entity: str
        self.type: int = 1
        self.det = set()
        self.copula_attr = set()
        self.sub_entity = set()
        self.attributes = set()
        self.actions = set()
        self.subj_relations = defaultdict(set)
        self.obj_relations = defaultdict(set)
        self.copula_attr_parent = set()
        self.sub_entity_parent = set()

class SceneGraph():
    def __init__(self):
        self.id_count: int = 0
        self.entities = dict()
        self.relations = defaultdict(set)
        self.waiting_list = list()
        self.question_ent = set()

    def delete_entity(self, id):
        ent = self.entities[id]
        for i in list(ent.copula_attr_parent):
            try:
                self.entities[i].copula_attr.remove(id)
            except ValueError:
                pass
        for i in list(ent.sub_entity_parent):
            try:
                self.entities[i].sub_entity.remove(id)
            except ValueError:
                pass
        for key, value in ent.subj_relations.items():
            try:
                self.entities[key[1]].obj_relations[key] -= value
            except KeyError:
                pass
            try:
                self.relations[key] -= value
            except KeyError:
                pass  
        for key, value in ent.obj_relations.items():
            try:
                self.entities[key[0]].subj_relations[key] -= value
            except KeyError:
                pass
            try:
                self.relations[key] -= value
            except KeyError:
                pass

        try:
            self.waiting_list.remove(id)
        except ValueError:
            pass

        del self.entities[id]

    def merge(self, id1, id2):
        if id1 == id2:
            return
        try:
            ent1 = self.entities[id1]
            ent2 = self.entities[id2]
        except KeyError:
            return

        merge_set= set()

        ent1.attributes = set(ent1.attributes).union(set(ent2.attributes))
        ent1.actions = set(ent1.actions).union(set(ent2.actions))

        subj_relations = dict()
        for key, value in ent1.subj_relations.items():
            for v in value:
                subj_relations[(' '.join(e[0] for e in v), self.entities[key[1]].id)] = self.entities[key[1]].entity
        obj_relations = dict()
        for key, value in ent1.subj_relations.items():
            for v in value:
                obj_relations[(' '.join(e[0] for e in v), self.entities[key[0]].id)] = self.entities[key[0]].entity

        for key, value in ent2.subj_relations.items():
            try: 
                obj = self.entities[key[1]]
            except KeyError:
                continue
            vi = None
            for v in value:
                for vi, ve in subj_relations.items():
                    if ' '.join(e[0] for e in v) == vi[0] and obj.entity == ve and vi[1] != key[1]:
                        obj.obj_relations[key].remove(v)
                        break
            if not obj.subj_relations and vi:
                merge_set.add((vi[1], key[1]))
            else:
                self.relations[(id1, key[1])] = value
                ent1.subj_relations[(id1, key[1])] = value

        for key, value in ent2.obj_relations.items():
            try: 
                subj = self.entities[key[0]]
            except KeyError:
                continue
            vi = None
            for v in value:
                for vi, ve in obj_relations.items():
                    if ' '.join(e[0] for e in v) == vi[0] and subj.entity == ve and vi[1] != key[0]:
                        subj.subj_relations[key].remove(v)
                        break
            if not subj.subj_relations and vi:
                merge_set.add(vi[1], key[0])
            else:
                self.relations[(key[0], id1)] = value
                ent1.obj_relations[(key[0], id1)] = value

        ent1.copula_attr = set(ent1.copula_attr).union(set(ent2.copula_attr))
        ent1.sub_entity = set(ent1.sub_entity).union(set(ent2.sub_entity))
        ent1.copula_attr_parent = set(ent1.copula_attr_parent).union(set(ent2.copula_attr_parent))
        ent1.sub_entity_parent = set(ent1.sub_entity_parent).union(set(ent2.sub_entity_parent))

        self.delete_entity(id2)
        for i in merge_set:
            self.merge(i[0], i[1])

    def add_to_wait(self, ent):
        if ent.entity in self.question_ent:
            return
        flag = True
        for i in range(len(self.waiting_list)):
            if self.entities[self.waiting_list[i]].sense_score > ent.sense_score:
                self.waiting_list.insert(i, ent.id)
                self.question_ent.add(ent.entity)
                flag = False
                break
        if flag:
            self.waiting_list.append(ent.id)
            self.question_ent.add(ent.entity)

    def raise_question(self):
        if self.waiting_list:
            return self.entities[self.waiting_list.pop(0)].entity
        else:
            return None
        
    def visualize(self, filename, notebook=False):
        network = Network(directed=True)

        def add_copula_attr():
            network.add_node(ent.id, label=ent.entity, size = 50, color=color)
            for copula_attr in list(ent.copula_attr):
                network.add_node(copula_attr, label=self.entities[copula_attr].entity, size = 30,  shape='star', color=color)
                network.add_edge(ent.id, copula_attr, color=color)
            for sub_entity in list(ent.sub_entity):
                network.add_node(sub_entity, label=self.entities[sub_entity].entity, size = 30, color=color)
                network.add_edge(ent.id, sub_entity, color=color)

        for ent in self.entities.values():
            if ent.type == 1:
                color='#ffaeb9'
                add_copula_attr()
            elif ent.type == -1:
                color='#ffa500'
                add_copula_attr()

            color = '#a8dda8'
            for attr in list(ent.attributes):
                attr_id = str(ent.id) + attr
                network.add_node(attr_id, label=attr, color=color)
                network.add_edge(ent.id, attr_id, color=color)

            color = '#bfefff'
            for action in list(ent.actions):
                act_id = str(ent.id) + action
                network.add_node(act_id, label=action, shape='diamond', color=color)
                network.add_edge(ent.id, act_id, color=color)

        color = '#bfefff'
        for ent_ids, relats in self.relations.items():
            for relat in relats:
                relat = ' '.join(e[0] for e in relat)
                relat_id = str(ent_ids) + relat
                network.add_node(relat_id, label=relat, color=color)
                subj_id = ent_ids[0]
                obj_id = ent_ids[1]
                network.add_edge(subj_id, relat_id, color=color)
                network.add_edge(relat_id, obj_id, color=color)
            
        network.toggle_physics(True)
        network.show(filename+'.html', notebook=notebook)