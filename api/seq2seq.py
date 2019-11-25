import logging
import datetime, time
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import numpy as np
import sentencepiece as spm

logger = logging.getLogger("api")
logger.info(tf.VERSION)

class Seq2Seq:
    def __init__(self):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load("/home/ubuntu/work/m.model")
        self.enc_export_path = '/home/ubuntu/work/export_enc'
        self.dec_export_path = '/home/ubuntu/work/export_dec'
        self.MAX_LENGH = 170
        # 一度に複数回predictすると落ちるのでその対策
        self.predicting = False

    def str_to_tokens(self, sentence : str ):
        ids = self.sp.EncodeAsIds(sentence)
        array = np.zeros(self.MAX_LENGH, dtype='int')
        for i, id in enumerate(ids):
            array[i] = id
        array = np.expand_dims(array, axis=0)
        return array

    def predict(self, text):
        self.predicting = True
        logger.info("predict: " + text)
        start_time = datetime.datetime.now()

        enc_input_data = self.str_to_tokens(text)
        states_values = []
        with tf.Session(graph=tf.Graph()) as sess:
            model_x = tf.saved_model.loader.load(sess, [tag_constants.SERVING], self.enc_export_path)
            sig_def = model_x.signature_def['encoder_states']
            enc_in_name = sig_def.inputs['enc_inp'].name
            enc_state_h_name = sig_def.outputs['encoder_state_h'].name
            enc_state_c_name = sig_def.outputs['encoder_state_c'].name
            result = sess.run([enc_state_h_name, enc_state_c_name],
                    feed_dict={enc_in_name: enc_input_data})
            states_values = [result[0], result[1]]

        logger.info(states_values)
 
        empty_target_seq = np.zeros( ( 1 , self.MAX_LENGH ) )
        empty_target_seq[0, 0] = self.sp.PieceToId('<s>')
            
        decoded_translation = ''
        with tf.Session(graph=tf.Graph()) as sess:
            model_x = tf.saved_model.loader.load(sess, [tag_constants.SERVING], self.dec_export_path)
            sig_def = model_x.signature_def['predict_output']
            dec_in_name = sig_def.inputs['dec_inp'].name
            dec_state_h_name = sig_def.inputs['dec_state_h'].name
            dec_state_c_name = sig_def.inputs['dec_state_c'].name
            out_name = sig_def.outputs['decoder_output'].name
            out_decoder_state_h_name = sig_def.outputs['decoder_state_h'].name
            out_decoder_state_c_name = sig_def.outputs['decoder_state_c'].name
            stop_condition = False
            while not stop_condition :
                result = sess.run([out_name, out_decoder_state_h_name, out_decoder_state_c_name],
                        feed_dict={dec_in_name:empty_target_seq, dec_state_h_name:states_values[0], dec_state_c_name:states_values[1]})
                
                sampled_word_index = np.argmax( result[0][0, -1, :] )
                sampled_word = None
                word = self.sp.IdToPiece(int(sampled_word_index)).replace('▁', ' ')
                decoded_translation += ' {}'.format( word )
                sampled_word = word

                if sampled_word == '</s>' or len(decoded_translation.split()) > 10:
                    stop_condition = True
                    break 
                    
                empty_target_seq = np.zeros( ( 1 , self.MAX_LENGH ) )  
                empty_target_seq[ 0 , 0 ] = sampled_word_index
                states_values = [ result[1] , result[2] ]
        
        logger.info("result: " + decoded_translation)
        logger.info("total: " + str(datetime.datetime.now() - start_time) + "s")
        self.predicting = False

        return decoded_translation.replace('</s>', '')

    def isPredicting(self):
        return self.predicting