"""text -> spike patterns for the ear input.

The model's ear has N_CH topographic channels (frequency bands).  Text is turned
into a stream of sparse channel patterns:

  * each token gets a reproducible set of channels
  * tokens that share context share channels (character n-gram hashing), so
    related words are physically similar in the input -- which is what lets the
    topographic auditory layer see them as related
  * a token is presented for a duration that scales with its length, with a
    short inter-token gap, so the input looks like speech rather than a list

Nothing here is learned or symbolic inside the model: the encoder is the
"cochlea", outside the brain.
"""
import hashlib
import numpy as np

N_CH = 280
TOK_CH = 40          # channels lit per token
MS_PER_CHAR = 45.0   # presentation length scales with the word length
GAP_MS = 40.0


def _h(s):
    return int.from_bytes(hashlib.blake2b(s.encode('utf-8'), digest_size=8).digest(), 'big')


def token_channels(tok, n_ch=N_CH, k=TOK_CH, grams=(1, 2, 3)):
    """channels for one token: a union of character n-gram codes.

    Shared character n-grams (e.g. 'cat' and 'cats') therefore share channels,
    which is the input-side analogue of morphological similarity.
    """
    tok = ' ' + tok.lower() + ' '
    chans = set()
    for g in grams:
        for i in range(max(1, len(tok) - g + 1)):
            h = _h(tok[i:i + g] + str(g))
            rng = np.random.default_rng(h)
            take = max(2, k // (12 * g))
            chans.update(rng.choice(n_ch, take, replace=False).tolist())
    if len(chans) < 6:
        rng = np.random.default_rng(_h(tok))
        chans.update(rng.choice(n_ch, 12, replace=False).tolist())
    return np.array(sorted(chans))


def pattern(tok, n_ch=N_CH):
    p = np.zeros(n_ch, dtype=np.float32)
    p[token_channels(tok, n_ch)] = 1.0
    return p


def stream(text, rate_hz=60.0, n_ch=N_CH):
    """yield (duration_ms, pattern) for each token of `text`"""
    out = []
    for tok in text.split():
        dur = max(80.0, MS_PER_CHAR * len(tok))
        out.append((dur, pattern(tok, n_ch) * rate_hz))
        out.append((GAP_MS, np.zeros(n_ch, dtype=np.float32)))
    return out


def similarity(a, b):
    ca, cb = set(token_channels(a)), set(token_channels(b))
    return len(ca & cb) / len(ca | cb)


if __name__ == '__main__':
    words = ['cat', 'cats', 'dog', 'the', 'sat', 'mat', 'on', 'the', 'cat', 'sat']
    print('channels per token: %s' % [(w, len(token_channels(w))) for w in words[:4]])
    print('\npairwise input similarity (shared / union):')
    pairs = [('cat', 'cats'), ('cat', 'dog'), ('cat', 'mat'), ('the', 'cat'), ('sat', 'cat')]
    for a, b in pairs:
        print('  %-5s vs %-5s : %.2f' % (a, b, similarity(a, b)))
    s = stream('the cat sat on the mat')
    total = sum(d for d, _ in s)
    print('\n"the cat sat on the mat" -> %d segments, %.0f ms total' % (len(s), total))
    print('first 6 durations: %s' % ['%.0f' % d for d, _ in s[:6]])
