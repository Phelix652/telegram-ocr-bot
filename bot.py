import os
import base64
import requests
import asyncio
from PIL import Image, ImageEnhance, ImageFilter
from google import genai
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes

# 🔹 YOUR TOKENS

import os

BOT_TOKEN = os.getenv("BOT_TOKEN")

GOOGLE_VISION_API_KEY = os.getenv("GOOGLE_VISION_API_KEY")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# 🔹 Safety Check
if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN not set")

if not GOOGLE_VISION_API_KEY:
    raise ValueError("GOOGLE_VISION_API_KEY not set")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not set")

client = genai.Client(api_key=GEMINI_API_KEY)

VISION_URL = f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_VISION_API_KEY}"

# 🔥 Resize Image
def preprocess_image(path):
    img = Image.open(path)

    if img.mode != "RGB":
        img = img.convert("RGB")

    base_width = 1800
    w_percent = base_width / float(img.size[0])
    h_size = int(float(img.size[1]) * w_percent)

    img = img.resize((base_width, h_size), Image.LANCZOS)

    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=160))

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)  # reduced contrast (more stable OCR)

    img.save(path, format="JPEG")
    return path

# 🔥 Split Long Image (WITH OVERLAP)
def split_image(path, chunk_height=1800, overlap=200):
    img = Image.open(path)
    width, height = img.size

    if height <= 2200:
        return [path]

    chunks = []
    y = 0
    count = 0

    while y < height:
        box = (0, y, width, min(y + chunk_height, height))
        cropped = img.crop(box)

        chunk_path = f"chunk_{count}.jpg"
        cropped.save(chunk_path)
        chunks.append(chunk_path)

        y += chunk_height - overlap
        count += 1

    return chunks

# 🔥 Google Vision OCR (DOCUMENT MODE)
def google_ocr(image_path):
    with open(image_path, "rb") as img_file:
        content = base64.b64encode(img_file.read()).decode()

    body = {
        "requests": [
            {
                "image": {"content": content},
                "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
                "imageContext": {
                    "languageHints": ["en", "my"]
                }
            }
        ]
    }

    try:
        response = requests.post(VISION_URL, json=body, timeout=30)
        result = response.json()
    except Exception as e:
        print("VISION REQUEST FAILED:", e)
        return []

    if "error" in result:
        print("VISION ERROR:", result["error"])
        return []

    try:
        full_text = result["responses"][0]["fullTextAnnotation"]["text"]

        # Split into lines
        lines = [line.strip() for line in full_text.split("\n") if line.strip()]

        return lines

    except:
        return []

def translate_batch(sentences):
    try:
        import json

        prompt = (
            "You are a professional comic translator.\n"
            "Translate each line into Myanmar.\n"
            "Return STRICT JSON array.\n"
            "Same number of items as input.\n"
            "If unsure, return empty string for that index.\n"
            "Do not explain.\n\n"
        )

        prompt += json.dumps(sentences, ensure_ascii=False)

        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=prompt,
            config={"response_mime_type": "application/json"}
        )

        # 🔥 Safe extraction
        if hasattr(response, "text") and response.text:
            raw_text = response.text.strip()
        else:
            try:
                raw_text = response.candidates[0].content.parts[0].text.strip()
            except Exception as e:
                print("FAILED TO EXTRACT RESPONSE:", e)
                return []

        print("RAW GEMINI:", raw_text)

        # 🔥 Remove markdown safely
        if raw_text.startswith("```"):
            raw_text = raw_text.replace("```json", "").replace("```", "").strip()

        try:
            translated = json.loads(raw_text)
        except Exception as e:
            print("JSON PARSE ERROR:", e)
            return []

        return translated

    except Exception as e:
        print("GEMINI ERROR:", e)
        return []

# 🔥 Telegram Handler (OPTIMIZED)
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    processing_msg = await update.message.reply_text("⏳ Processing image...")

    file_path = "input.jpg"

    if update.message.photo:
        photo = update.message.photo[-1]
        file = await photo.get_file()
        await file.download_to_drive(file_path)

    elif update.message.document:
        doc = update.message.document
        if not doc.mime_type.startswith("image/"):
            await update.message.reply_text("❌ Please send an image file.")
            return
        file = await doc.get_file()
        await file.download_to_drive(file_path)
    else:
        await update.message.reply_text("❌ Please send an image.")
        return

    preprocess_image(file_path)
    chunks = split_image(file_path)

    # 🔥 PARALLEL OCR
    async def ocr_chunk(chunk):
        return await asyncio.to_thread(google_ocr, chunk)

    tasks = [ocr_chunk(chunk) for chunk in chunks]
    results = await asyncio.gather(*tasks)

    sentences = []
    for block_list in results:
        sentences.extend(block_list)

    if not sentences:
        await processing_msg.edit_text("❌ No text detected.")
        return

    translated_sentences = await asyncio.to_thread(
        translate_batch,
        sentences
    )

    if not translated_sentences:
        await processing_msg.edit_text("❌ Translation failed.")
        return

    # Auto-fix length mismatch
    if len(translated_sentences) < len(sentences):
        translated_sentences += [""] * (len(sentences) - len(translated_sentences))

    translated_sentences = translated_sentences[:len(sentences)]

    reply = ""
    for en, mm in zip(sentences, translated_sentences):
        reply += f"“{en}”\n→ “{mm}”\n\n"

    await processing_msg.edit_text(reply[:4000])

    # 🔥 CLEANUP FILES
    try:
        os.remove(file_path)
    except:
        pass

    for chunk in chunks:
        try:
            os.remove(chunk)
        except:
            pass

# 🔥 Start Bot
app = ApplicationBuilder().token(BOT_TOKEN).build()

app.add_handler(
    MessageHandler(
        filters.PHOTO | filters.Document.IMAGE,
        handle_photo
    )
)

print("🤖 Bot Running...")
app.run_polling()
