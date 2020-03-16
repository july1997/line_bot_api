from django.db import models
import hashlib

class Question(models.Model):
    hash_text = models.CharField(max_length=128)
    reply_text = models.CharField(max_length=183)

    def __str__(self):
        return self.reply_text

def get_reply(text):
    # MD5ハッシュ化
    hs = hashlib.md5(text.encode()).hexdigest()
    # DB検索
    return Question.objects.filter(hash_text=hs)

def save_reply(text, reply_text):
     # MD5ハッシュ化
    hs = hashlib.md5(text.encode()).hexdigest()
    
    q = Question(hash_text=hs, reply_text=reply_text)
    q.save()
