from django.test import TestCase

# Create your tests here.
from api.seq2seq import Seq2Seq_with_attention

class ModelTests(TestCase):

    def test_seq2seq_with_attention(self):
        model = Seq2Seq_with_attention()
        predicted = model.predict('こんにちは')
        print(predicted)
        self.assertIsNot(len(predicted), 0)