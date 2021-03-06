import logging
import datetime, time
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from tensorflow.keras.models import load_model
import sentencepiece as spm

logger = logging.getLogger("api")
logger.info(tf.VERSION)

class Seq2Seq:
    def __init__(self):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load("/home/ubuntu/work/m.model")
        self.MAX_LENGTH = 170
        self.export_path = '/home/ubuntu/work/model'
        self.sess = tf.compat.v1.keras.backend.get_session()
        self.model = tf.saved_model.loader.load(self.sess, [tag_constants.SERVING], self.export_path)
        self.sig_def = self.model.signature_def['predict_output']
        self.in_enc_name = self.sig_def.inputs['enc_input'].name
        self.in_dec_name = self.sig_def.inputs['dec_input'].name
        self.out_name = self.sig_def.outputs['output'].name
        # 一度に複数回predictすると落ちるのでその対策
        self.predicting = False

    def str_to_tokens(self, sentence : str ):
        ids = self.sp.EncodeAsIds(sentence)
        array = np.zeros(self.MAX_LENGTH, dtype='int')
        for i, id in enumerate(ids):
            array[i] = id
        array = np.expand_dims(array, axis=0)
        return array

    def predict(self, text):
        self.predicting = True
        logger.info("predict: " + text)
        start_time = datetime.datetime.now()

        encoder_input = self.str_to_tokens(text)
        decoder_input = np.zeros( ( 1 , self.MAX_LENGTH ) )
        decoder_input[0, 0] = self.sp.PieceToId('<s>')
        for i in range(1, 10):
            output = self.sess.run(self.out_name,
                feed_dict={self.in_enc_name: encoder_input, self.in_dec_name: decoder_input}).argmax(axis=2)
            decoder_input[:,i] = output[:,i]

        decoded_translation = ""
        for embed in decoder_input[0,1:]:
            word = self.sp.IdToPiece(int(embed)).replace('▁', ' ').replace('<unk>', ' ').replace(' ', ' ')
            decoded_translation += ' {}'.format(word)
        
        logger.info("result: " + decoded_translation)
        logger.info("total: " + str(datetime.datetime.now() - start_time) + "s")
        self.predicting = False

        return decoded_translation.replace('</s>', '')

    def isPredicting(self):
        return self.predicting