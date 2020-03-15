from django.db import models

class Question(models.Model):
    hash_text = models.CharField(max_length=128)
    reply_text = models.CharField(max_length=183)

    def __str__(self):
        return self.reply_text
