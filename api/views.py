from api.seq2seq import *
from api.apikey import LINE_CHANNEL_SECRET, LINE_CHANNEL_ACCESS_TOKEN
from django.http import HttpResponseForbidden, HttpResponse
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError, LineBotApiError
from linebot.models import (
    MessageEvent, TextMessage, FollowEvent, UnfollowEvent,
    TextSendMessage, ImageMessage, AudioMessage
)
# csrf 検証無効化
from django.views.decorators.csrf import csrf_exempt

import logging
logger = logging.getLogger("api")

# 各クライアントライブラリのインスタンス作成
line_bot_api = LineBotApi(channel_access_token=LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(channel_secret=LINE_CHANNEL_SECRET)

# Seq2Seq with Attention モデルへ変更(Seq2Seq モデルと同時に読み込むとOOMエラーになるのでやらないこと)
model = Seq2Seq_with_attention()

# csrf 検証を無効化したい関数に @csrf_exempt を設定
@csrf_exempt
def callback(request):
    # signatureの取得
    signature = request.META['HTTP_X_LINE_SIGNATURE']
    body = request.body.decode('utf-8')
    try:
        # 署名の検証を行い、成功した場合にhandleされたメソッドを呼び出す
        handler.handle(body, signature)
    except LineBotApiError as e:
        print("Got exception from LINE Messaging API: %s\n" % e.message)
        for m in e.error.details:
            print("  %s: %s" % (m.property, m.message))
        print("\n")
    except InvalidSignatureError as e:
        print("InvalidSignatureError: %s\n" % e.message)
        return HttpResponseForbidden()
    return HttpResponse('OK')

# フォローイベントの場合の処理
@handler.add(FollowEvent)
def handle_follow(event):
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text='初めまして！いっぱい話しかけてね。')
    )

# メッセージイベントの場合の処理
@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    text = ""
    if not model.isPredicting():
        text = model.predict(text=event.message.text)
    else:
        return HttpResponseForbidden()
        
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=text)
    )

@csrf_exempt
def chat(request):
    logger.info(request.POST['text'])
    text = request.POST['text']
    if not text is None and not model.isPredicting():
        return HttpResponse(model.predict(text=text), content_type="text/plain")
    return HttpResponseForbidden()