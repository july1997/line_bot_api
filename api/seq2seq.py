import logging
import datetime, time
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from tensorflow.keras.models import load_model
import sentencepiece as spm
import api.normalizer as normalizer
from api.models import get_reply, save_reply

logger = logging.getLogger("api")
logger.info(tf.__version__)

class Inference():
    def __init__(self):
        # 一度に複数回predictすると落ちるのでその対策
        self.predicting = False

    def predict(self, text):
        self.predicting = True
        logger.info("predict: " + text)
        start_time = datetime.datetime.now()

        # 文章の正規化
        text = normalizer.twitter_normalizer(text.replace("\n", ""))
        logger.info("normalize: " + text)

        # DBからリプライを検索
        req = get_reply(text)
        if len(req) == 0:
            reply = self.model_predict(text)
            save_reply(text, reply)
            logger.info("save: " + text + " : " +  reply)
        else:
            reply = req[0].reply_text
            logger.info("load: " + reply)

        logger.info("result: " + reply)
        logger.info("total: " + str(datetime.datetime.now() - start_time) + "s")
        self.predicting = False
        return reply

    def model_predict(self, text):
        return ""

    def isPredicting(self):
        return self.predicting

class Seq2Seq(Inference):
    def __init__(self):
        super().__init__()
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load("/home/ubuntu/work/seq2seq/m.model")
        self.MAX_LENGTH = 170
        self.export_path = '/home/ubuntu/work/seq2seq/model'
        self.sess = tf.compat.v1.keras.backend.get_session()
        self.model = tf.compat.v1.saved_model.loader.load(self.sess, [tag_constants.SERVING], self.export_path)
        self.sig_def = self.model.signature_def['predict_output']
        self.in_enc_name = self.sig_def.inputs['enc_input'].name
        self.in_dec_name = self.sig_def.inputs['dec_input'].name
        self.out_name = self.sig_def.outputs['output'].name

    def str_to_tokens(self, sentence : str ):
        ids = self.sp.EncodeAsIds(sentence)
        array = np.zeros(self.MAX_LENGTH, dtype='int')
        for i, id in enumerate(ids):
            array[i] = id
        array = np.expand_dims(array, axis=0)
        return array

    def model_predict(self, text):
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

        return decoded_translation.replace('</s>', '')

class Seq2Seq_with_attention(Inference):
    def __init__(self):
        super().__init__()
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load("/home/ubuntu/work/attention/m.model")
        self.MAX_LENGTH = 183
        self.loaded = tf.saved_model.load('/home/ubuntu/work/attention')
        self.inference_func = self.loaded.signatures["serving_default"]
        logger.info(self.inference_func.structured_outputs)
        
    def str_to_tokens(self, sentence : str ):
        ids = [1] + self.sp.EncodeAsIds(sentence) + [2]
        array = np.zeros(self.MAX_LENGTH, dtype='int')
        for i, id in enumerate(ids):
            array[i] = id
        array = np.expand_dims(array, axis=0)
        return array

    def model_predict(self, text):                
        encoder_input = self.str_to_tokens(text)
        decoder_input = np.zeros( ( 1 , self.MAX_LENGTH ) )
        decoder_input[0, 0] = self.sp.PieceToId('</s>')
        for i in range(1, self.MAX_LENGTH-1):
            output = self.inference_func(
                    inputs=tf.constant(encoder_input, dtype=tf.float32), dec_inputs=tf.constant(decoder_input, dtype=tf.float32))
            output = output['outputs'].numpy().argmax(axis=-1)
            decoder_input[:,i] = output[:,i-1]
            if output[:,i-1] == 1:
                break

        decoded_translation = ""
        outputs = tf.cast(np.array(decoder_input), tf.int32).numpy().tolist()
        for output in outputs:
            dec = self.sp.DecodeIds(list(reversed([s for s in output if s != 0])))
            decoded_translation = dec.replace('<unk>', '').replace('⁇', '').replace('▁', ' ')

        return decoded_translation.replace('</s>', '').replace('<mention>', '')
    
    def isPredicting(self):
        return False
        
