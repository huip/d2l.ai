from mxnet import nd
import random
import zipfile


with zipfile.ZipFile('./data/jaychou_lyrics.txt.zip') as zin:
    with zin.open('jaychou_lyrics.txt') as f:
        corpus_chars = f.read().decode('utf-8')


corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
corpus_chars = corpus_chars[0:10000]

idx_to_char = list(set(corpus_chars))

char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])

vocab_size = len(char_to_idx)

corpus_index = [char_to_idx[char] for char in corpus_chars]


sample = corpus_index[:20]

print('chars:', ''.join([idx_to_char[idx] for idx in sample]))
print('indices:', sample)


def data_iter_random(corpus_indices, batch_size, num_steps, ctx=None):
    num_examples = (len(corpus_indices) - 1) // num_steps
    epoch_size = num_examples // batch_size
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)

    def _data(pos):
        return corpus_index[pos: pos + num_steps]

    for i in range(epoch_size):
        i = i * batch_size
        batch_indices = example_indices[i: i+batch_size]
        X = [_data(j * num_steps) for j in batch_indices]
        Y = [_data(j * num_steps + 1) for j in batch_indices]
        yield nd.array(X, ctx), nd.array(Y, ctx)

# my_seq = list(range(30))
# for X, Y in data_iter_random(my_seq, batch_size=2, num_steps=6):
#     print('X: ', X, '\nY:', Y, '\n')


def data_iter_consecutive(corpus_indices, batch_size, num_steps, ctx=None):
    corpus_indices = nd.array(corpus_indices, ctx=ctx)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size
    indices = corpus_indices[0: batch_size*batch_len].reshape((
        batch_size, batch_len))

    epoch_size = (batch_len - 1) // num_steps

    for i in range(epoch_size):
        i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i + 1: i + num_steps + 1]
        yield X, Y

my_seq = list(range(30))
for X, Y in data_iter_consecutive(my_seq, batch_size=2, num_steps=6):
    print('X: ', X, '\nY:', Y, '\n')
