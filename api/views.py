from api.seq2seq import Seq2Seq
from django.http import HttpResponseForbidden, HttpResponse
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError, LineBotApiError
from linebot.models import (
    MessageEvent, TextMessage, FollowEvent, UnfollowEvent,
    TextSendMessage, ImageMessage, AudioMessage
)
# csrf 検証無効化
from django.views.decorators.csrf import csrf_exempt

channel_secret = os.getenv('LINE_CHANNEL_SECRET', None)
channel_access_token = os.getenv('LINE_CHANNEL_ACCESS_TOKEN', None)

# 各クライアントライブラリのインスタンス作成
line_bot_api = LineBotApi(channel_access_token=channel_access_token)
handler = WebhookHandler(channel_secret=channel_secret)

# Seq2Seq モデル
model = Seq2Seq()

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
    
    text = model.predict(text=event.message.text)

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=text)
    )

