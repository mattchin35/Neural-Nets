import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import rnn
import time
import collections
from random import shuffle
import pickle
tf.reset_default_graph()
st = time.clock()


class VariableLengthRnn:

    def __init__(self, alpha, k):
        fileroot = '/Users/matthewchin/Documents/Kaggle_toxic/'
        trainroot = fileroot + 'train.csv'
        testroot = fileroot + 'test.csv'
        sample = fileroot + 'sample_submission.csv.zip'

        # data can be accessed as a series with df.iloc[idx]
        # series categories are id, comment_text, toxic, severe_toxic,
        # obscene, threat, insult, identity_hate
        # Task is to classify the comments according to each of their
        # given labels, and use that to score a test set. Output is one-hot.

        self.train_df = pd.read_csv(trainroot, header=0, sep=',')
        self.test_df = pd.read_csv(testroot, header=0, sep=',')

        self.alpha = alpha
        if k == "full" or k == 0:
            self.k = self.train_df.shape[0]
        else:
            self.k = k

        (self.fwd_dict, _), self.comments = \
            self.preprocess(self.train_df, self.k)
        (test_dict, _), self.test_comments = self.preprocess(self.test_df, self.test_df.shape[0])
        self.fwd_dict.update(test_dict)
        self.train_tuples, self.test_inputs, self.maxlen = \
            self.make_tuples(self.train_df, self.comments, self.test_comments)
        self.n = len(self.train_tuples)
        self.c = 6

        """
        Parameters
        """

        self.dh = 512  # number of hidden layer nodes
        self.alpha = alpha  # learning rate

        # read in a previously created set of weights
        weights = pd.read_csv('toxic_weights.csv', sep=',', index_col=0).values
        bias = pd.read_csv('toxic_bias.csv', sep=',', index_col=0).values
        self.weights = {
            'out': tf.Variable(weights.astype(np.float32))
        }
        self.biases = {
            'out': tf.Variable(bias.astype(np.float32))
        }

        # self.weights = {
        #     'out': tf.Variable(tf.random_normal([self.dh, self.c]))
        # }
        # self.biases = {
        #     'out': tf.Variable(tf.random_normal([self.c]))
        # }


    @staticmethod
    def build_dataset(words):
        """
        This function builds a forwards and backwards dictionary.
        The words in the dictionary are sorted by frequency from
        high (0, first) to low (900+, last).
        :param words:
        :return:
        """
        count = collections.Counter(words).most_common()
        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary)
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        # fwd_dict: word to number code
        # rev_dict: number code to word
        return dictionary, reverse_dictionary

    @staticmethod
    def length(x):
        """
        From https://danijar.com/variable-sequence-lengths-in-tensorflow/.
        Find the length of a sequence on the fly for variable-length
        sequence processing.
        :param x: Input sequence to find the length of.
        :return: length, duh
        """
        used = tf.sign(tf.reduce_max(tf.abs(x), 2))
        length = tf.reduce_sum(used, 1)
        length = tf.cast(length, tf.int32)
        return length

    @staticmethod
    def last_relevant(output, length):
        """
        From https://danijar.com/variable-sequence-lengths-in-tensorflow/.
        Get the last relevant output from a padded sequence of outputs
        for variable-length sequence processing.
        :param output: tensor of outputs from the input sequence (batch x time x dh)
        :param length: scalar of the unpadded sequence length
        :return:
        """
        batch_size = tf.shape(output)[0]  # batch size
        max_length = tf.shape(output)[1]  # max seq length
        out_size = int(output.get_shape()[2])  # hidden layer size
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, out_size])  # flatten to 2d
        relevant = tf.gather(flat, index)  # get desired output vectors
        return relevant

    def make_tuples(self, train_df, train_com, test_com):
        """
        Make tuples to pass through RNN.
        :param df: dataframe
        :return:
        """
        # Make words lowercase, remove weird
        # symbols and text encodings.
        labels = train_df.loc[:, 'toxic':]
        labels = labels.values
        # n, c = labels.shape

        """
        Sentences must be converted to numbers. Batch size is along the 
        x-axis, 'timesteps' are along the y-axis (dim 1), and data 
        dimensionality is on the z axis. A commented example below shows 
        how.
        """
        # com = comments[0]
        # sentence_as_num = np.array([fwd_dict[word] for word in com])
        # sentence_as_num = np.reshape(sentence_as_num, [-1, len(com), 1])

        # put everything in this form
        purify = lambda comments: [np.array([self.fwd_dict[word] for word in com]).reshape(-1, len(com), 1)
                                   for com in comments if com != []]

        train_sentence = purify(train_com)
        test_sentence = purify(test_com)

        train_len = np.amax(np.array([A.shape[1] for A in train_sentence]))
        testlen = np.amax(np.array([A.shape[1] for A in test_sentence]))
        maxlen = np.amax([train_len, testlen])

        # pad all the vectors to the max length for processing
        train_sentence = [self.pad_vector(x, maxlen) for x in train_sentence]
        test_sentence = [self.pad_vector(x, maxlen) for x in test_sentence]

        # shuffle the training labels and sentences together
        tuples = [(train_sentence[i], labels[i, :]) for i in range(len(train_sentence))]
        # shuffle(tuples)  # in place shuffle - object is changed (ignore during testing)
        tuples = tuples[:self.k]

        return tuples, test_sentence, maxlen

    @staticmethod
    def pad_vector(x, max):
        if x.shape[1] < max:
            return np.concatenate([x, np.zeros((1, max-x.shape[1], 1))], axis=1)
        else:
            return x

    def preprocess(self, df, sz):
        """
        Preprocess the data from its pandas dataframe.
        :param df: dataframe
        """

        # 2-layer list flattening
        flatten = lambda l: [item for sublist in l for item in sublist]
        comments = sz * [None]
        for i in range(sz):  # either self.k or df.shape[0] here
            # train_df['comment_text'][i]
            temp = df['comment_text'][i]
            temp = temp.lower()
            temp = temp.splitlines()
            temp = " ".join(temp)
            temp = temp.split()
            temp = [word.split('/') for word in temp]
            temp = flatten(temp)
            temp = [word.split('-') for word in temp]
            temp = flatten(temp)
            words = [word.strip('.-:=?,;!)(/_[]"') for word in temp]
            # comments.append(words)
            comments[i] = words

        all_words = flatten(comments)
        # vocab_size = len(all_words)

        return self.build_dataset(all_words), comments

    def RNN(self, x, weights, biases):
        # Data must be prepared to match rnn function requirements
        # reshape to [1, n_input]
        # n_input = x.shape[1]
        xlen = self.length(x)
        lstm_cell = rnn.BasicLSTMCell(self.dh)
        # for a dynamic rnn, do not unstack the inputs as the
        # word prediction lstm example does
        # batching can be done with dynamic_partition, ignored here
        outputs, states = tf.nn.dynamic_rnn(lstm_cell, x,
                                            dtype=tf.float32,
                                            sequence_length=xlen)

        last = self.last_relevant(outputs, xlen)

        # get activation
        return tf.matmul(last, weights['out']) + biases['out']

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    vrnn = VariableLengthRnn(.001, 0)

    toxic_count = np.sum(np.array([np.amax(label) for (com, label) in vrnn.train_tuples]))

    # n_input = 1  # sentence fragment length
    # for variable length sentences, leave timesteps as none
    X = tf.placeholder("float", [None, vrnn.maxlen, 1])
    Y = tf.placeholder("float", [None, vrnn.c])

    """
    Constructing the model and optimization using the placeholders
    """
    logits = vrnn.RNN(X, vrnn.weights, vrnn.biases)
    prediction = tf.sigmoid(logits)

    # It seems RMSProp may be better here - the TF tutorial uses
    # regular gradient descent.
    loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=logits, labels=Y))
    optimizer = tf.train.RMSPropOptimizer(learning_rate=vrnn.alpha)
    train_op = optimizer.minimize(loss_op)

    # evaluation for multilabel classification
    rnd = tf.round(prediction)
    correct_pred = tf.cast(tf.equal(rnd, Y), tf.float32)
    accuracy = tf.reduce_mean(correct_pred)  # cast bool to float

    mid = time.clock()
    print('Setup time: {}'.format(mid - st))

    # Run the training
    train_split = vrnn.train_tuples[:int(vrnn.n * .8)]
    n_train = len(train_split)

    test_split = vrnn.train_tuples[int(vrnn.n * .8):]
    n_test = len(test_split)
    test_iter = iter(test_split)

    # create an initializer
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    num_epochs = 5
    display_step = 1e3
    save_step = 1e4
    print('Number of toxic training comments: {}'.format(toxic_count))
    if True:
        for epoch in range(num_epochs):
            acc_total = 0
            loss_total = 0
            shuffle(train_split)
            train_iter = iter(train_split)  # reinitialize iterator
            print('Epoch {}'.format(epoch+1))

            for step in range(n_train):
                batch_x, batch_y = next(train_iter)
                if len(batch_y.shape) == 1:
                    batch_y = batch_y.reshape(-1, batch_y.shape[0])

                _, loss, pred, acc = sess.run([train_op, loss_op, rnd, accuracy],
                                              feed_dict={X: batch_x, Y: batch_y})

                if (step+1) % display_step == 0:
                    print('Step {}'.format(step+1))
                    print("Iter= " + str(step + 1) + ", Average Loss= " +
                          "{:.6f}".format(loss_total / (step+1)) + ", Average Accuracy= " +
                          "{:.2f}%".format(100 * acc_total / (step+1)))

                    # print("Accuracy: {}".format(acc))
                    # print('pred: {}'.format(pred.astype(int)))
                    # print('real: {}'.format(batch_y))

                if (step + 1) % save_step == 0:
                    print('Saving...')
                    W = sess.run(vrnn.weights['out'])
                    B = sess.run(vrnn.biases['out'])
                    pd.DataFrame(W).to_csv('toxic_weights.csv')
                    pd.DataFrame(B).to_csv('toxic_bias.csv')

                loss_total += loss
                acc_total += acc

            # Epoch results
            print("Epoch " + str(epoch+1) + ", Average Loss= " +
                  "{:.6f}".format(loss_total / n_train) + ", Average Accuracy= " +
                  "{:.2f}%".format(100 * acc_total / n_train))

            W = sess.run(vrnn.weights['out'])
            B = sess.run(vrnn.biases['out'])
            pd.DataFrame(W).to_csv('toxic_weights.csv')
            pd.DataFrame(B).to_csv('toxic_bias.csv')

    # testing with a validation set rather than kaggle's test set
    print('Testing')
    display_step = 1e3
    acc_total = 0
    for step in range(n_test):
        batch_x, batch_y = next(test_iter)
        if len(batch_y.shape) == 1:
            batch_y = batch_y.reshape(-1, batch_y.shape[0])

        pred, acc = sess.run([rnd, accuracy], feed_dict={X: batch_x, Y: batch_y})
        if (step + 1) % display_step == 0:
            print("Iter= " + str(step + 1) + ", Average Accuracy= " +
                  "{:.2f}%".format(100 * acc_total / (step+1)))

        acc_total += acc

    # Epoch results
    print("Iter= " + str(step + 1) + ", Total Accuracy= " +
          "{:.2f}%".format(100 * acc_total / n_test))

    fin = time.clock()
    print('Eval time: {}'.format(fin-mid))



