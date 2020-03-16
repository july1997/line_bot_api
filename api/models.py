from django.db import models
import hashlib

class Question(models.Model):
    hash_text = models.CharField(max_length=128)
    reply_text = models.CharField(max_length=183)

    def __str__(self):
        return self.reply_text

def FastReply(text, reply_text):
    # MD5ハッシュ化
    hs = hashlib.md5(text.encode()).hexdigest()
    # DB検索
    q = Question.objects.get(hash_text=hs)
    if len(q) == 0:
        # DB保存
        nq = Question(hash_text=hs, reply_text=reply_text)
        nq.save()
    else:
        return q.reply_text
    return ""
