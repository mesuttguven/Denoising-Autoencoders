"""
Models, batching and a manual training loop.

A manual loop (train_on_batch + best-weights early stopping) is used
instead of Sequence/fit so the code behaves the same across TF/Keras
versions.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


# ------------------------------------------------------------ architectures
def build_dae(size, arch="paper"):
    """
    arch="paper": the paper's architecture (max-pool bottleneck, no skips).
    arch="unet" : same widths plus skip connections at full resolution, so it
                  CAN reproduce pixel-level detail.  This is the best case for
                  the DAE: if even this does not help, the conclusion is not an
                  artefact of a weak autoencoder.
    """
    if arch == "paper":
        m = models.Sequential([
            layers.Input(shape=(size, size, 3)),
            layers.Conv2D(32, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(2, padding="same"),
            layers.Conv2D(64, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(2, padding="same"),
            layers.Conv2D(128, 3, activation="relu", padding="same"),
            layers.UpSampling2D(2),
            layers.Conv2D(64, 3, activation="relu", padding="same"),
            layers.UpSampling2D(2),
            layers.Conv2D(32, 3, activation="relu", padding="same"),
            layers.Conv2D(3, 3, activation="sigmoid", padding="same"),
        ])
    elif arch == "unet":
        L = layers
        conv = lambda f: L.Conv2D(f, 3, activation="relu", padding="same")
        inp = L.Input(shape=(size, size, 3))
        c1 = conv(32)(conv(32)(inp))
        c2 = conv(64)(conv(64)(L.MaxPooling2D(2)(c1)))
        b = conv(128)(conv(128)(L.MaxPooling2D(2)(c2)))
        d2 = conv(64)(conv(64)(L.Concatenate()([L.UpSampling2D(2)(b), c2])))
        d1 = conv(32)(conv(32)(L.Concatenate()([L.UpSampling2D(2)(d2), c1])))
        m = models.Model(inp, L.Conv2D(3, 1, activation="sigmoid")(d1))
    else:
        raise ValueError(arch)
    m.compile(optimizer="adam", loss="mse")
    return m


class Blur:
    """
    Non-learned low-pass baseline: k x k average pooling, then bilinear
    upsampling back.  Has the same call interface as a Keras model so it can
    be dropped in wherever a DAE is used.  If a DAE does no better than this,
    it is acting as a blur filter, not as a de-obfuscator.
    """
    def __init__(self, k=4):
        self.k = k

    def predict_on_batch(self, x):
        y = tf.nn.avg_pool2d(x, self.k, self.k, "VALID")
        return tf.image.resize(y, x.shape[1:3], method="bilinear").numpy()


def build_cnn(kind, size):
    L = layers
    if kind == "cnn1":      # paper's CNN Model 1: 3 conv + dense64
        stack = [L.Conv2D(32, 3, activation="relu", padding="same"), L.MaxPooling2D(2),
                 L.Conv2D(64, 3, activation="relu", padding="same"), L.MaxPooling2D(2),
                 L.Conv2D(128, 3, activation="relu", padding="same"), L.MaxPooling2D(2),
                 L.Flatten(), L.Dense(64, activation="relu")]
    elif kind == "cnn2":    # deeper, dropout-regularised
        stack = []
        for f in (32, 64, 128):
            stack += [L.Conv2D(f, 3, activation="relu", padding="same"),
                      L.Conv2D(f, 3, activation="relu", padding="same"),
                      L.MaxPooling2D(2), L.Dropout(0.25)]
        stack += [L.Flatten(), L.Dense(128, activation="relu"), L.Dropout(0.5),
                  L.Dense(64, activation="relu")]
    else:
        raise ValueError(kind)
    m = models.Sequential([L.Input(shape=(size, size, 3))] + stack
                          + [L.Dense(1, activation="sigmoid")])
    m.compile(optimizer=tf.keras.optimizers.Adam(3e-4), loss="binary_crossentropy")
    return m


# ------------------------------------------------------------------ batching
def batch_iter(store, idx, sources, batch, rng, target=None, dae=None,
               shuffle=True):
    """
    Yield (x, y).

    sources : list of store keys.  With several, each sample draws one of
              its VALID sources at random (re-drawn every call, i.e. every
              epoch) -- this is how augmentation / the union DAE mix
              clean and obfuscated versions.
    target  : None -> y = class label;  'clean' -> y = clean image (DAE).
    dae     : optional Keras model applied to x (normalisation step).
    """
    idx = np.asarray(idx)
    if shuffle:
        idx = rng.permutation(idx)
    for s in range(0, len(idx), batch):
        b = np.sort(idx[s:s + batch])
        if len(sources) == 1:
            src = np.zeros(len(b), int)
        else:
            V = np.stack([store.valid[n][b] for n in sources], 1)
            scores = rng.random(V.shape)
            scores[~V] = -1.0
            src = scores.argmax(1)
        x = np.empty((len(b), store.size, store.size, 3), np.float32)
        for k, name in enumerate(sources):
            m = src == k
            if m.any():
                x[m] = store.arrays[name][b[m]] / 255.0
        if dae is not None:
            x = np.clip(dae.predict_on_batch(x), 0.0, 1.0).astype(np.float32)
        if target is None:
            y = store.labels[b].astype(np.float32)
        else:
            y = store.arrays[target][b].astype(np.float32) / 255.0
        yield x, y


def predict_scores(model, store, idx, source, dae=None, batch=128):
    out = []
    for x, _ in batch_iter(store, idx, [source], batch, None, dae=dae,
                           shuffle=False):
        out.append(np.asarray(model.predict_on_batch(x)).reshape(-1))
    return np.concatenate(out) if out else np.zeros(0)


# ------------------------------------------------------------------ training
def _scalar(r):
    return float(np.asarray(r).reshape(-1)[0])


def fit(model, train_batches, val_batches, epochs, patience, tag=""):
    """train_batches(): generator factory; val_batches(): generator factory."""
    best, best_w, wait = np.inf, None, 0
    for ep in range(epochs):
        tl = [_scalar(model.train_on_batch(x, y)) for x, y in train_batches()]
        vl = [_scalar(model.test_on_batch(x, y)) for x, y in val_batches()]
        tl, vl = float(np.mean(tl)), float(np.mean(vl))
        print(f"    {tag} epoch {ep + 1}/{epochs} train={tl:.4f} val={vl:.4f}")
        if vl < best - 1e-5:
            best, best_w, wait = vl, model.get_weights(), 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_w is not None:
        model.set_weights(best_w)
    return model
