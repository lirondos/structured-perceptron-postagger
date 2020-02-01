class Sentence(object):
    def __init__(self, snt):
        self.snt = snt
        return

    def features(self, sentence, i):
        features = set()
        features.add("curr="+sentence.snt[i][0])  # curr word
        features.add("pref="+sentence.snt[i][0][:4])  # pref
        features.add("suff="+sentence.snt[i][0][-4:]) # suff
        if i < len(sentence.snt) - 2:  # next next word
            features.add("nn_="+sentence.snt[i + 2][0])
        if i < len(sentence.snt) - 1:  # 1 next word
            features.add("n="+sentence.snt[i + 1][0])
        if i > 1:  # prev prev word
            features.add("pp="+sentence.snt[i - 2][0])
        if i > 0:  # prev word
            features.add("p="+sentence.snt[i - 1][0])
        return features


class Trellis:
    def __init__(self, tagset_length, sentence_length):
        self.columns = [Column(tagset_length) for j in range(sentence_length)]


class Column:
    def __init__(self, tagset_length):
        self.states = [None for j in range(tagset_length)]

class State:
    def __init__(self, my_feature_vector, score, j, z):
        self.feature_vector = my_feature_vector
        self.score = score
        self.current_tag = j
        self.prev_state = z

