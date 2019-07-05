import numpy as np
import pandas as pd
import tensorflow as tf
import re
import pickle
from gensim.models import Word2Vec

class CharRNN:
    """
    RNN to generate text based on input text files. Base on code from
    https://r2rt.com/recurrent-neural-networks-in-tensorflow-ii.html.
    """

    def __init__(self, files=[], newfiles=True, loadfile='word_dict.pckl', save_name='model'):
        # Character level processing
        self.embedding_size = 30
        self.save_name = save_name
        if newfiles:
            word_dict = self.new_files(files, name=save_name)
        else:
            word_dict = self.load_file(loadfile)

        self.text = word_dict['text']
        self.sentences = word_dict['sentences']
        self.words = word_dict['words']
        self.word_to_ix = word_dict['word_to_ix']
        self.ix_to_word = word_dict['ix_to_word']
        self.state_size = 512
        # self.seq_length = 25
        self.learning_rate = 1e-4
        self.data = [None] * len(self.sentences)
        for i in range(len(self.sentences)):
            self.data[i] = [self.word_to_ix[word] for word in self.sentences[i]]

        self.prepare_w2v()

        self.data_len = len(self.data)
        self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')
        self.num_steps = 100
        self.num_classes = len(self.words)
        epsilon = np.sqrt(6) / np.sqrt(self.state_size + self.num_classes)
        self.W = tf.Variable(tf.random_uniform([self.state_size, self.num_classes],
                                               -epsilon, epsilon), name='W')
        self.inputs, self.targets = self.prepare_training_data()
        # self.embeddings = tf.Variable(tf.one_hot(range(self.num_classes), depth=range(self.num_classes)),
        #                               name='embedding_matrix')
        self.graph = self.build_graph()

        train_dataset = tf.data.Dataset.from_tensor_slices((self.graph['feat_ph'], self.graph['label_ph']))
        # train_dataset = train_dataset.shuffle(int(1E6)).batch(tf.cast(self.batch_size, tf.int64))  # .repeat()  # inferred repeat
        train_dataset = train_dataset.shuffle(int(1E6)).batch(tf.cast(self.batch_size, tf.int64)).repeat()  # inferred repeat
        self.train_iter = train_dataset.make_initializable_iterator()
        self.next_element = self.train_iter.get_next()

    def new_files(self, files, name='new_txt'):
        """
        Read the files to be processed, and make a single file with the text combined.
        files: a list of file paths.
        """
        raw = []
        for file in files:
            with open(file) as f:
                raw.append(f.read())

        text = '\n\n'.join(raw)
        goodchar = '[^A-Za-z0-9\n. ]+'
        text = re.sub(goodchar, '', text)
        text = text.lower()
        text = re.sub('[.]+', ' <SENTENCE-END> ', text)
        sentences = text.splitlines()
        sentence_lists = [s.split() for s in sentences]
        words = [word for s in sentence_lists for word in s]
        words = list(set(words))

        with open(name + '.txt', 'w+') as f:
            f.write(text)

        word_to_ix = {ch: i for i, ch in enumerate(words)}
        ix_to_word = {i: ch for i, ch in enumerate(words)}

        word_dict = dict(
            text=text,
            sentences=sentence_lists,
            words=words,
            word_to_ix=word_to_ix,
            ix_to_word=ix_to_word,
        )

        with open(name + '_word_dict.pckl', 'wb') as f:
            pickle.dump(word_dict, f)

        return word_dict

    def load_file(self, file):
        with open(file, 'rb') as f:
            word_dict = pickle.load(f)

        return word_dict

    def prepare_w2v(self):
        model = Word2Vec(self.sentences, size=150)
        print(model.wv.vectors)
        print(model.wv.index2word)
        print(model.wv.word_vec())

        # model.save("shake_wv.bin")

    def prepare_training_data(self):
        """
        Prepare inputs and targets as (n_samples x n_steps) matrices
        :return: inputs, targets
        """
        idx = 0
        inputs = []
        targets = []
        while idx < self.data_len:
            end = min(self.num_steps, self.data_len - idx)
            inputs.append(self.data[idx: idx + end])
            targets.append(self.data[idx + 1:idx + end + 1])
            idx += end

        if min(len(inputs[-1]), len(targets[-1])) < self.num_steps:
            # just get rid of the last weird one - sufficient data already present
            del inputs[-1]
            del targets[-1]

        inputs = np.array(inputs)
        targets = np.array(targets)
        return inputs, targets

    def create_embeddings(self, word2vec=False):
        if word2vec:  # skip-gram embeddings
            embeddings = tf.Variable(tf.random_uniform([self.num_classes, self.embedding_size], -1.0, 1.0))
        else:
            # simplest embedding version
            embeddings = tf.Variable(tf.one_hot(range(self.num_classes), depth=range(self.num_classes)),
                                     name='embedding_matrix')

        return embeddings

    def build_graph(self):
        # x is a set of inputs, each row of which is for a single piece of data
        # each data contains a separate index for the word or character being referenced
        x = tf.placeholder(tf.int32, [None, None], name='input_placeholder')
        y = tf.placeholder(tf.int32, [None, self.num_steps], name='labels_placeholder')

        embeddings = tf.get_variable('embedding_matrix', [self.num_classes, self.state_size])  # auto initializes
        inputs = tf.nn.embedding_lookup(embeddings, x)

        cell = tf.nn.rnn_cell.LSTMCell(self.state_size, state_is_tuple=True)
        # for multicell rnn
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * 3, state_is_tuple=True)
        init_state = cell.zero_state(self.batch_size, tf.float32)
        rnn_outputs, state = tf.nn.dynamic_rnn(cell, inputs, initial_state=init_state, dtype=tf.float32)  # outputs are (batch_size x timesteps x state_size)
        # rnn_outputs, state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)

        W = tf.get_variable('W', [self.state_size, self.num_classes])
        b = tf.get_variable('b', [self.num_classes], initializer=tf.constant_initializer(0.0))
        outputs = tf.reshape(rnn_outputs, [-1, self.state_size])  # reshape to ((batch_size x timesteps) x state_size)
        y_reshaped = tf.reshape(y, [-1])  # reshape tensor from 2d with timesteps to 1d, all integer labels in a line

        logits = tf.matmul(outputs, W) + b  # ((batch_size x timesteps) x n_classes)
        predictions = tf.nn.softmax(logits)
        maxpred = tf.cast(tf.argmax(predictions, axis=1), tf.int32) # ((batch size x timesteps) x 1)
        accuracy = tf.reduce_sum(tf.cast(tf.equal(maxpred, y_reshaped), tf.float32))

        # use one_hot and y_reshaped to [-1, num_classes] for non-sparse softmax
        # one_hot = tf.one_hot(y, depth=self.num_classes, axis=-1, dtype=tf.float32)  # tensor (batch x n_steps x n_classes)
        y_reshaped = tf.reshape(y, [-1])  # reshape tensor from 2d with timesteps to 1d, all integer labels in a line
        loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped))
        train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss_op)

        feat_ph = tf.placeholder(self.inputs.dtype, self.inputs.shape)
        label_ph = tf.placeholder(self.targets.dtype, self.targets.shape)

        graph = dict(x=x,
                     y=y,
                     feat_ph=feat_ph,
                     label_ph=label_ph,
                     init_state=init_state,
                     final_state=state,
                     loss=loss_op,
                     train_op=train_op,
                     pred=predictions,
                     acc=accuracy,
                     saver=tf.train.Saver())

        return graph

    def run_training(self, n_epochs=1000, checkpoint=None, resume=False):
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        if type(checkpoint) == str and resume:
            self.graph['saver'].restore(sess, checkpoint)

        display_step = int(20)
        save_step = int(1e2)
        step = 0
        batch_size = 32
        n_train = int(1e3)

        print("Starting...")
        # before loop for infinite dataset, after for epochs
        sess.run(self.train_iter.initializer, feed_dict={self.graph['feat_ph']: self.inputs,
                                                         self.graph['label_ph']: self.targets,
                                                         self.batch_size: batch_size})
        for epoch in range(n_epochs):
            # sess.run(self.train_iter.initializer, feed_dict={self.graph['feat_ph']: self.inputs,
            #                                                  self.graph['label_ph']: self.targets,
            #                                                  self.batch_size: batch_size})

            acc_total = 0
            loss_total = 0
            for _ in range(n_train):
                try:
                    # print('step {}'.format(step))
                    batch_x, batch_y = sess.run(self.next_element)
                    if batch_x.shape[0] < batch_size:
                        continue

                    # print("step {}".format(step+1), batch_x.shape, batch_y.shape)
                    # batch_x = batch_x.squeeze()
                    # batch_y = batch_y.squeeze()
                    nd = np.prod(batch_x.shape)
                    # if (step+1) < 220:
                    #     continue

                    _, loss, acc = sess.run([self.graph['train_op'], self.graph['loss'], self.graph['acc']],
                                            feed_dict={self.graph['x']: batch_x, self.graph['y']: batch_y,
                                                       self.batch_size: batch_size})
                    acc_total += acc
                    loss_total += loss

                    # stuff to show progress - deal with this next week
                    if (step + 1) % display_step == 0:
                        print("Iter = " + str(step + 1) + ", Loss= " +
                              "{:.6f}".format(loss / nd) + ", Accuracy= " +
                              "{:.2f}%".format(acc / nd))

                    if (step + 1) % save_step == 0:
                        print('Saving...')
                        save_path = self.graph['saver'].save(sess, "./" + self.save_name, global_step=step)

                    step += 1
                except tf.errors.OutOfRangeError:
                    print('Epoch {}'.format(epoch+1))
                    print('Average Loss = {:.6f}'.format(loss_total / self.data_len))
                    print("Average Accuracy = {:.2f}%".format(acc_total / self.data_len))
                    print('Saving...')
                    save_path = self.graph['saver'].save(sess, "./" + self.save_name, global_step=step)
                    break

        sess.close()

    def generate_characters(self, checkpoint, num_chars, prompt='A', pick_top_chars=None):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.graph['saver'].restore(sess, checkpoint)

            state = None

            current_char = self.char_to_ix[prompt]
            chars = [current_char]
            # words = [self.ix_to_char[c] for c in chars]
            # print(current_char)

            for i in range(num_chars):
                if state:
                    feed_dict = {self.graph['x']: [[current_char]], self.graph['init_state']: state, self.batch_size: 1}
                else:
                    feed_dict = {self.graph['x']: [[current_char]], self.batch_size: 1}

                # print(feed_dict.keys())
                pred, state = sess.run([self.graph['pred'], self.graph['final_state']], feed_dict=feed_dict)

                if pick_top_chars:
                    p = np.squeeze(pred)
                    p[np.argsort(p)[:-pick_top_chars]] = 0
                    p = p / np.linalg.norm(p)
                    current_char = np.random.choice(self.num_classes, 1, p=p)[0]
                else:
                    current_char = np.random.choice(self.num_classes, 1, p=np.squeeze(pred))[0]

                # print(current_char)
                chars.append(current_char)

            # print(chars)
            words = [self.ix_to_char[c] for c in chars]
            print(''.join(words))


if __name__ == "__main__":
    # kdot = 'kdot_lyrics/'
    # kdot_albums = ['section_80.txt', 'gkmc.txt', 'tpab.txt', 'untitled_unmastered.txt', 'DAMN.txt']
    home = '/Users/matthewchin/'
    files = [home+'Documents/shakespeare_3mb.txt']
    rnn = CharRNN(files=files, newfiles=True, save_name='w2v_test', loadfile='w2v_test_word_dict.pckl')
    # print(rnn.chars)
    train = True
    if train:
        pass
        # checkpoint = "model-8999"
        # rnn.run_training(n_epochs=int(1e2), checkpoint=checkpoint, resume=False)
    else:
        pass
        # load_file = "shake_model_abr-47099"
        # rnn.generate_characters(checkpoint=load_file, num_chars=10000)


