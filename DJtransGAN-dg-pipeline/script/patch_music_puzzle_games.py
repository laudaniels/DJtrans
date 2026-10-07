"""Port the vendored music_puzzle_games/model.py (SEN) from TF1 tf.contrib to
modern TensorFlow, so the pretrained checkpoint can still be loaded and run.

tf.contrib was removed entirely in TensorFlow 2.x. `tf_slim` is the official
standalone continuation of tf.contrib.layers/slim and is API-compatible, so
the model only needs tf.contrib.layers.* swapped for tf_slim equivalents
(plus one tf.layers call that Keras 3 dropped). The checkpoint's variable
names are unchanged, so the pretrained weights still load correctly.

Run this once, after `git clone https://github.com/remyhuang/music-puzzle-games.git
music_puzzle_games` (see this repo's README):

    python script/patch_music_puzzle_games.py

Verified to reproduce the original author's reference output
(music_puzzle_games/output/output.csv) to ~4 significant digits.
"""
import os

PATCHED_SOURCE = '''import tensorflow.compat.v1 as tf
import tf_slim as slim

class SEN():
    def __init__(self, is_train):
        self.is_train = is_train
        self.bn_params = {'is_training': self.is_train,
                          'center': True, 'scale': True,
                          'updates_collections': None}
        self.build_model()

    def init_place_holder(self):
        self.x1 = tf.placeholder(tf.float32, shape=[None, None, 128, 1])
        self.x2 = tf.placeholder(tf.float32, shape=[None, None, 128, 1])
        self.y = tf.placeholder(tf.float32, shape=[None, 2])

    def conv(self, inputs, filters, kernel, stride):
        return slim.conv2d(inputs, filters, kernel, stride,
                                        activation_fn=tf.nn.relu,
                                        normalizer_fn=slim.batch_norm,
                                        normalizer_params=self.bn_params)

    def fc(self, inputs, num_units, act=tf.nn.relu):
        keep_rate = 0.5
        _inputs = slim.dropout(inputs, keep_rate, is_training=self.is_train)
        return slim.fully_connected(_inputs, num_units,
                                                 activation_fn=act,
                                                 normalizer_fn=slim.batch_norm,
                                                 normalizer_params=self.bn_params)

    def cnn(self, inputs, name, reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            h = self.conv(inputs, 128, [4, 128], [1, 128])
            h = self.conv(h, 256, [4, 1], [1, 1])
            h = self.conv(h, 256, [4, 1], [1, 1])
        return h

    def reduce_var(self, inputs, axis):
        m = tf.reduce_mean(inputs, axis=axis, keep_dims=True)
        devs_squared = tf.square(inputs - m)
        return tf.reduce_mean(devs_squared, axis=axis)

    def build_model(self):
        # placeholder
        self.init_place_holder()

        # Early Conv.
        h1 = tf.squeeze(self.cnn(self.x1, 'cnn'), axis=2)
        h2 = tf.squeeze(self.cnn(self.x2, 'cnn', reuse=True), axis=2)

        # Consine Similarity
        num = tf.matmul(h1, h2, transpose_b=True)
        h1_norm = tf.sqrt(tf.reduce_sum(tf.square(h1), axis=2, keep_dims=True))
        h2_norm = tf.sqrt(tf.reduce_sum(tf.square(h2), axis=2, keep_dims=True))
        denom =  tf.matmul(h1_norm, h2_norm, transpose_b=True)
        fms = tf.expand_dims(tf.div(num, denom), 3)

        # Late Conv.
        h = self.conv(fms, 64, [3, 3], [1, 1])
        h = slim.max_pool2d(h, [3, 3], stride=[3, 3], padding='SAME')
        h = self.conv(h, 128, [3, 3], [1, 1])
        h = slim.max_pool2d(h, [3, 3], stride=[3, 3], padding='SAME')
        h = self.conv(h, 256, [3, 3], [1, 1])

        # Global Pooling
        g_max = tf.reduce_max(h, [1, 2])
        g_avg = tf.reduce_mean(h, [1, 2])
        g_var = self.reduce_var(h, [1, 2])
        h = tf.concat([g_max, g_avg, g_var], axis=1)

        # Classifier
        h = self.fc(h, 1024, act=tf.nn.relu)
        h = self.fc(h, 1024, act=tf.nn.relu)
        logits = self.fc(h, 2, act=None)
        self.predictions = tf.nn.softmax(logits)
        self.loss = tf.losses.softmax_cross_entropy(self.y, logits)
        self.loss = tf.reduce_mean(self.loss)

        # Train
        if self.is_train:
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            self.train_op = optimizer.minimize(self.loss)
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)

    def calculate(self, sess, batch):
        feed_dict = {self.x1: batch.x1,
                     self.x2: batch.x2,
                     self.y: batch.y}
        predictions = sess.run(self.predictions, feed_dict=feed_dict)
        return predictions
'''

path = os.path.join(os.path.dirname(__file__), '..', 'music_puzzle_games', 'model.py')
path = os.path.normpath(path)

if not os.path.exists(path):
    raise SystemExit(
        f'{path} not found - clone music_puzzle_games first (see README): '
        'git clone https://github.com/remyhuang/music-puzzle-games.git music_puzzle_games'
    )

current = open(path, encoding='utf-8').read()
if current == PATCHED_SOURCE:
    print(f'Already patched: {path}')
else:
    open(path, 'w', encoding='utf-8').write(PATCHED_SOURCE)
    print(f'Patched: {path}')

# main.py's standalone demo (not used by the pipeline, but handy for a quick
# `python main.py` sanity check) passes the signal positionally; newer
# librosa made melspectrogram's arguments keyword-only.
main_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'music_puzzle_games', 'main.py'))
if os.path.exists(main_path):
    old_call = 'librosa.feature.melspectrogram(y, sr=22050, n_fft=2048, hop_length=512, n_mels=128)'
    new_call = 'librosa.feature.melspectrogram(y=y, sr=22050, n_fft=2048, hop_length=512, n_mels=128)'
    main_src = open(main_path, encoding='utf-8').read()
    if new_call in main_src:
        print(f'Already patched: {main_path}')
    elif old_call in main_src:
        open(main_path, 'w', encoding='utf-8').write(main_src.replace(old_call, new_call))
        print(f'Patched: {main_path}')
    else:
        print(f'Note: expected melspectrogram call not found in {main_path}, leaving it as-is')
