"""Check minibatch streaming against ordinary weighted Keras array training."""
import unittest
import numpy as np
import tensorflow as tf
from streaming import install_streaming_fit, batched_dataset


class StreamingTests(unittest.TestCase):
    def test_weighted_training_and_validation_across_epochs(self):
        x=np.arange(48,dtype=np.float32).reshape(12,4)/48
        y=np.tile(np.arange(4,dtype=np.int64),3)
        def build():
            m=tf.keras.Sequential([tf.keras.Input((4,)),tf.keras.layers.Dense(
                4,kernel_initializer='zeros',bias_initializer='zeros')])
            m.compile(optimizer=tf.keras.optimizers.SGD(.01),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
            return m
        original=tf.keras.Model.fit
        try:
            baseline=build()
            kwargs=dict(batch_size=4,epochs=2,shuffle=False,
                        class_weight={0:1.,1:2.,2:3.,3:4.},verbose=0)
            baseline.fit(x,y,**kwargs)
            install_streaming_fit()
            streamed=build()
            history=streamed.fit(x,y,validation_data=(x,y),**kwargs)
            for a,b in zip(baseline.get_weights(),streamed.get_weights()):
                np.testing.assert_allclose(a,b,rtol=1e-6,atol=1e-7)
            self.assertEqual(len(history.history['val_loss']),2)
        finally:
            tf.keras.Model.fit=original

    def test_subject_inputs_and_partial_final_batch(self):
        x={'eeg':np.zeros((5,3,2),np.float32),'subject_id':np.arange(5,dtype=np.int32)}
        y=np.arange(5,dtype=np.int64)
        batches=list(batched_dataset(x,y,2,shuffle=False).as_numpy_iterator())
        self.assertEqual([len(b) for _,b in batches],[2,2,1])
        np.testing.assert_array_equal(np.concatenate([a['subject_id'] for a,_ in batches]),y)


if __name__=='__main__':
    unittest.main()
