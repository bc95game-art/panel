import os
import base64
import tempfile
from io import BytesIO
from flask import Flask, request, jsonify, render_template
from anthropic import Anthropic
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
anthropic_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# حداکثر تعداد پیام‌هایی که به API ارسال می‌شوند (کنترل هزینه و توکن)
MAX_HISTORY_TURNS = 10

@app.route('/')
def index():
    return render_template('index.html')

# ------------------------------------------------------------
# 1. آپلود فایل به Files API کلود (دریافت file_id)
# ------------------------------------------------------------
@app.route('/api/upload', methods=['POST'])
def upload_file():
    """دریافت فایل از کاربر، آپلود به Claude Files API و برگرداندن file_id"""
    if 'file' not in request.files:
        return jsonify({'error': 'فایلی ارسال نشده'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'نام فایل خالی است'}), 400
    
    # تشخیص MIME type بر اساس پسوند (ساده شده)
    mime_type = file.mimetype or 'application/octet-stream'
    
    # آپلود فایل با استفاده از SDK رسمی
    try:
        # ایجاد یک فایل موقتی برای ارسال به SDK
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        
        # بارگذاری به Claude Files API (نسخه بتا)
        with open(tmp_path, 'rb') as f:
            uploaded = anthropic_client.beta.files.upload(
                file=(file.filename, f, mime_type)
            )
        
        os.unlink(tmp_path)  # حذف فایل موقت
        
        return jsonify({
            'success': True,
            'file_id': uploaded.id,
            'filename': file.filename,
            'mime_type': mime_type
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ------------------------------------------------------------
# 2. تولید تصویر (DALL-E 3)
# ------------------------------------------------------------
@app.route('/api/generate-image', methods=['POST'])
def generate_image():
    data = request.json
    prompt = data.get('prompt', '').strip()
    if not prompt:
        return jsonify({'error': 'پرامپت نمی‌تواند خالی باشد'}), 400
    
    try:
        response = openai_client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        image_url = response.data[0].url
        return jsonify({'success': True, 'image_url': image_url})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ------------------------------------------------------------
# 3. پیام دادن به کلود (با پشتیبانی از فایل‌ها و تاریخچه)
# ------------------------------------------------------------
@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_text = data.get('text', '').strip()
    # تاریخچه به صورت لیست از پیام‌های قبلی (هر پیام شامل role و content)
    history = data.get('history', [])   # فرمت استاندارد Anthropic: [{"role": "user", "content": ...}, ...]
    file_id = data.get('file_id')       # اگر فایلی همراه پیام باشد
    filename = data.get('filename', '')
    
    # ساخت محتوای پیام کاربر (متن + فایل)
    content_blocks = []
    
    # اگر فایل وجود داشته باشد، بلوک document یا image اضافه می‌کنیم
    if file_id:
        # تشخیص نوع فایل از روی MIME (در مرحله آپلود ذخیره نشده، ساده می‌گیریم)
        # برای تصاویر باید از بلوک image استفاده کنیم، برای PDF/Office از بلوک document
        # در عمل بهتر است نوع فایل را هنگام آپلود ذخیره کنیم. برای سادگی فرض می‌کنیم فایل PDF است.
        # اما برای بهبود: می‌توانیم از filename پسوند را تشخیص دهیم.
        ext = filename.split('.')[-1].lower() if '.' in filename else ''
        if ext in ['jpg', 'jpeg', 'png', 'gif', 'webp']:
            content_blocks.append({
                "type": "image",
                "source": {
                    "type": "file",
                    "file_id": file_id
                }
            })
        else:
            content_blocks.append({
                "type": "document",
                "source": {
                    "type": "file",
                    "file_id": file_id
                },
                "citations": {"enabled": True}   # فعال کردن ارجاع به صفحات
            })
    
    if user_text:
        content_blocks.append({
            "type": "text",
            "text": user_text
        })
    
    # اگر محتوایی برای ارسال وجود نداشت
    if not content_blocks:
        return jsonify({'error': 'هیچ پیام یا فایلی ارسال نشده'}), 400
    
    # تاریخچه را به آخرین چند پیام محدود می‌کنیم
    limited_history = history[-MAX_HISTORY_TURNS*2:] if len(history) > MAX_HISTORY_TURNS*2 else history
    
    # ساخت لیست پیام‌ها برای API
    messages = limited_history.copy()
    messages.append({
        "role": "user",
        "content": content_blocks
    })
    
    try:
        # فراخوانی Claude Messages API
        response = anthropic_client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=4096,
            messages=messages,
            betas=["files-api-2025-04-14"]   # فعال کردن Files API بتا
        )
        
        reply_text = response.content[0].text
        
        # استخراج citations اگر وجود داشته باشد (اختیاری)
        citations = []
        for block in response.content:
            if hasattr(block, 'citations') and block.citations:
                citations.extend(block.citations)
        
        return jsonify({
            'success': True,
            'reply': reply_text,
            'citations': citations,
            'usage': {
                'input_tokens': response.usage.input_tokens,
                'output_tokens': response.usage.output_tokens
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
