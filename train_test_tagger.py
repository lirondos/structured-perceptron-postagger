import sys
from perceptron_pos_tagger import Perceptron_POS_Tagger
from data_structures import Sentence

def read_in_gold_data(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [[tup.split('_') for tup in line.split()] for line in lines]
        sents = [Sentence(line) for line in lines]

    return sents 


def read_in_plain_data(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.split() for line in lines]
        sents = [Sentence(line) for line in lines]

    return sents 


def output_auto_data(auto_data, filename):
    for sent in auto_data:
        with open(filename, "a") as f:
            to_write = [elem[0]+"_"+elem[1] for elem in sent.snt]
            f.write(" ".join(to_write) + "\n")


if __name__ == '__main__':

    # run this script to produce an annotated version of the test file
    # default parameters: 4 iterations, all features, averaged model
    train_file = "train/ptb_02-21.tagged"
    gold_dev_file = "dev/ptb_22.tagged"
    test_file = "test/ptb_23.snt"
    test_data = read_in_plain_data(test_file)
    train_data = read_in_gold_data(train_file)
    gold_dev_data = read_in_gold_data(gold_dev_file)
    my_tagger = Perceptron_POS_Tagger()
    my_tagger.train(train_data, gold_dev_data, average=True, to_be_ablated = [])
    auto_test_data = my_tagger.tag(test_data)
    output_auto_data(auto_test_data, "test/ptb_23.tagged")
