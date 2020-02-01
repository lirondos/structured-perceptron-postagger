from sparse_vector import Vector
import numpy as np
from data_structures import Trellis
from data_structures import Sentence
from data_structures import State
from tqdm import tqdm

ablation = []
class Perceptron_POS_Tagger(object):
    def __init__(self):
        self.tagset = None
        self.weights = Vector()

    def tag(self, test_data):
        results = []
        for sent in tqdm(test_data):
            add_slots = Sentence([[sent.snt[i], []] for i in range(len(sent.snt))])
            tag_sequence_predicted, predicted_feature_vector = self.viterbi(add_slots)
            new_sent = Sentence([[sent.snt[i], tag_sequence_predicted[i]] for i in range(len(sent.snt))])
            results.append(new_sent)
        return results

    def train(self, train_data, dev_data, average=False, to_be_ablated=None):
        if to_be_ablated:
            global ablation
            ablation = [to_be_ablated[0]]
        print("Training...")
        iterations = 4
        self.tagset = self.get_tagset(train_data)
        if average:
            batch = 100
            counter = 0
            big_predicted_feature_vector = Vector()
            big_gold_feature_vector = Vector()
            for iteration in range(iterations):
                for i in tqdm(range(len(train_data))):
                    sent = train_data[i]
                    tag_sequence_gold = [pair[1] for pair in sent.snt]
                    tag_sequence_predicted, predicted_feature_vector = self.viterbi(sent)
                    gold_feature_vector = self.get_gold_feature_vector(sent)
                    counter = counter + 1
                    if counter%batch == 0:
                        self.weights.__iadd__(big_gold_feature_vector)
                        self.weights.__isub__(big_predicted_feature_vector)
                        big_predicted_feature_vector = Vector()
                        big_gold_feature_vector = Vector()
                    else:
                        big_gold_feature_vector.__iadd__(gold_feature_vector)
                        big_predicted_feature_vector.__iadd__(predicted_feature_vector)
        else:
            for iteration in range(iterations):
                for i in tqdm(range(len(train_data))):
                    sent = train_data[i]
                    tag_sequence_gold = [pair[1] for pair in sent.snt]
                    tag_sequence_predicted, predicted_feature_vector = self.viterbi(sent)
                    gold_feature_vector = self.get_gold_feature_vector(sent)
                    self.weights.__iadd__(gold_feature_vector)
                    self.weights.__isub__(predicted_feature_vector)
        print("Getting accuracy on dev set...")
        acc = self.get_accuracy(dev_data)
        print(acc)

    def get_accuracy(self, dev_data):
        gold = list()
        predicted = list()
        for sent in tqdm(dev_data):
            sentence = [pair[0] for pair in sent.snt]
            tag_sequence_gold = [pair[1] for pair in sent.snt]
            tag_sequence_predicted, predicted_feature_vector = self.viterbi(sent)
            gold.extend(tag_sequence_gold)
            predicted.extend(tag_sequence_predicted)
        return sum(1 for x, y in zip(gold, predicted) if x == y) / len(gold)


    def get_tagset(self,train_data):
        tag_set = set()
        for sent in train_data:
            tags = [pair[1] for pair in sent.snt]
            tag_set.update(tags)
        tag_set = list(tag_set)
        tagset_dict = dict()
        for i in range(len(tag_set)):
            tagset_dict[i] = tag_set[i]
        return tagset_dict

    def create_feature_vector(self, features, curr_tag, prev_tag):
        my_vector = Vector()
        my_vector.v[("ptag="+prev_tag,curr_tag)] = 1 # prev tag
        for feature in features:
            my_vector.v[(feature, curr_tag)] = 1
        if ablation:
            key_to_ablated = None
            for k in my_vector.v.keys():
                if k[0].startswith(ablation[0]):
                    key_to_ablated = k
            if key_to_ablated:
                del my_vector.v[key_to_ablated]
        return my_vector

    def viterbi(self, sentence):
        trellis = Trellis(len(self.tagset), len(sentence.snt))  # trellis
        for i in range(len(trellis.columns)): # i is index of curr word
            column = trellis.columns[i]
            features = sentence.features(sentence, i)
            for j in range(len(column.states)): # j is index of curr tag
                if i==0:
                    my_feature_vector = self.create_feature_vector(features, self.tagset[j], "__START__")
                    score = self.weights.dot(my_feature_vector)
                    column.states[j] = State(my_feature_vector, score, self.tagset[j], None)
                else:
                    possible_states = []
                    prev_column = trellis.columns[i - 1]
                    for z in range(len(prev_column.states)): # z is index of prev tag
                        prev_state = prev_column.states[z]
                        my_feature_vector = self.create_feature_vector(features, self.tagset[j], self.tagset[z])
                        score = prev_state.score + self.weights.dot(my_feature_vector)
                        possible_states.append(State(my_feature_vector, score, self.tagset[j], prev_state))
                    possible_scores = np.array([state.score for state in possible_states])
                    index_best_score = possible_scores.argmax()
                    column.states[j] = possible_states[index_best_score]
        last_column = trellis.columns[len(sentence.snt)-1]
        last_column_scores = np.array([state.score for state in last_column.states])
        best_final_index = last_column_scores.argmax()
        best_final_state = last_column.states[best_final_index]
        tag_sequence_predicted, predicted_feature_vector = self.get_predicted_sequence(best_final_state)
        return tag_sequence_predicted, predicted_feature_vector


    def get_predicted_sequence(self, final_state):
        tag_seq = list()
        final_feature_vector = Vector()
        my_state = final_state
        while my_state:
            tag_seq.append(my_state.current_tag)
            final_feature_vector.__iadd__(my_state.feature_vector)
            my_state = my_state.prev_state
        tag_seq.reverse()
        return tag_seq, final_feature_vector

    def get_gold_feature_vector(self, sent):
        feature_vector_gold = Vector()
        for i in range(len(sent.snt)):
            features = sent.features(sent, i)
            if i == 0:
                local_feature = self.create_feature_vector(features, sent.snt[i][1], "__START__")
            else:
                local_feature = self.create_feature_vector(features, sent.snt[i][1], sent.snt[i-1][1])
            feature_vector_gold.__iadd__(local_feature)
        return feature_vector_gold