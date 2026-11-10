"""Bound TensorFlow host input copies to one batch for the SEED-IV run."""
import math
import numpy as np


def batched_dataset(x, y, batch_size, *, shuffle, seed=42):
    import tensorflow as tf
    n = len(y)
    def spec(a):
        return tf.TensorSpec((None, *a.shape[1:]), tf.as_dtype(a.dtype))
    signature = (tf.nest.map_structure(spec, x), spec(y))
    rng = np.random.default_rng(seed)
    def batches():
        indices = rng.permutation(n) if shuffle else np.arange(n)
        for start in range(0,n,batch_size):
            idx = indices[start:start+batch_size]
            yield tf.nest.map_structure(lambda a: a[idx], x), y[idx]
    ds = tf.data.Dataset.from_generator(batches, output_signature=signature)
    ds = ds.apply(tf.data.experimental.assert_cardinality(math.ceil(n/batch_size)))
    options = tf.data.Options()
    options.deterministic = True
    options.threading.private_threadpool_size = 2
    return ds.with_options(options).prefetch(1)


def install_streaming_fit():
    import tensorflow as tf
    original = tf.keras.Model.fit
    def fit(self, x=None, y=None, **kwargs):
        if y is not None and isinstance(y,np.ndarray):
            size = int(kwargs.pop('batch_size',32))
            shuffle = bool(kwargs.pop('shuffle',True))
            validation = kwargs.get('validation_data')
            if validation is not None:
                assert len(validation)==2, 'Expected explicit validation X/y'
                kwargs['validation_data'] = batched_dataset(*validation,size,shuffle=False)
            x = batched_dataset(x,y,size,shuffle=shuffle)
            print('STREAMING_FIT batches=',int(x.cardinality().numpy()),
                  'batch_size=',size,flush=True)
            y = None
        return original(self,x=x,y=y,**kwargs)
    tf.keras.Model.fit = fit
