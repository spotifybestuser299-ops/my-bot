# main.py
import os
import json
import uuid
import tempfile
import subprocess
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from gtts import gTTS
from supabase import create_client, Client

# --- Config from env ---
HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_API_KEY")
HUGGINGFACE_MODEL = os.environ.get("HUGGINGFACE_MODEL", "google/flan-t5-large")  # change if you prefer
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET", "ai_videos")
FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")  # ensure ffmpeg is installed in the runtime

if not all([HUGGINGFACE_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_KEY]):
    raise RuntimeError("Set HUGGINGFACE_API_KEY, SUPABASE_URL and SUPABASE_SERVICE_KEY env variables")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

app = FastAPI(title="Free AI Video Generator (FFmpeg + gTTS + HF)")

class GenerateRequest(BaseModel):
    topic: str
    role: str
    length_seconds: Optional[int] = 45  # how long approx the spoken script should be (guideline)

def call_hf_generate(topic: str, role: str, length_seconds: int = 45) -> dict:
    """
    Call HuggingFace inference API with a prompt that requests strict JSON:
    { "title": "...", "script": "...", "quiz": [ { "question": "...", "options": [...], "answer": "..." }, ... ] }
    """
    prompt = f"""
You are a witty, friendly teacher-bot. Create a short lesson for a {role} on the topic "{topic}".
Make it humorous (one clean joke or meme reference), conversational, and short enough to speak in about {length_seconds} seconds.
Return **only valid JSON** (no extra text) with keys:
  - title: short title
  - script: the lesson text (3-6 short sentences)
  - quiz: an array of 3 objects each with keys: question, options (array of 4 strings), answer (exact match of correct option)
Example:
{{"title":"...","script":"...","quiz":[{{"question":"...","options":["A","B","C","D"],"answer":"B" , ... ]}}
Make sure JSON is parseable.
"""
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}", "Accept": "application/json"}
    url = f"https://api-inference.huggingface.co/models/{HUGGINGFACE_MODEL}"
    payload = {"inputs": prompt, "options": {"wait_for_model": True, "use_cache": False}}
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    if not resp.ok:
        raise HTTPException(status_code=502, detail=f"HuggingFace error: {resp.status_code} {resp.text[:300]}")
    # HF sometimes returns text blobs; try to extract JSON from response
    out = resp.json()
    # The HF JSON response shape may vary by model; common format: [{"generated_text": "..."}]
    text = None
    if isinstance(out, list) and len(out) > 0:
        if isinstance(out[0], dict):
            # model may return {"generated_text": "..."}
            if "generated_text" in out[0]:
                text = out[0]["generated_text"]
            elif "text" in out[0]:
                text = out[0]["text"]
            else:
                # fallback: stringifying first element
                text = json.dumps(out[0])
        else:
            text = str(out[0])
    elif isinstance(out, dict) and "generated_text" in out:
        text = out["generated_text"]
    else:
        text = str(out)

    # Extract first JSON object from text
    try:
        # Find the first '{' and parse until balancing '}' — robust extraction
        start = text.find('{')
        if start == -1:
            raise ValueError("No JSON object found in model output")
        # naive balancing
        depth = 0
        end = None
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end is None:
            raise ValueError("Could not find end of JSON in model output")
        json_text = text[start:end+1]
        data = json.loads(json_text)
    except Exception as e:
        # As fallback, try to parse entire text
        try:
            data = json.loads(text)
        except Exception:
            raise HTTPException(status_code=502, detail=f"Could not parse JSON from model output: {str(e)} | raw: {text[:1000]}")
    # Validate keys
    if "script" not in data or "title" not in data or "quiz" not in data:
        raise HTTPException(status_code=502, detail=f"Invalid output from model: missing keys. raw: {json.dumps(data)[:800]}")
    return data

def tts_save(script_text: str, out_path: str) -> str:
    tts = gTTS(script_text)
    tts.save(out_path)
    return out_path

def make_video(audio_path: str, title: str, output_path: str, duration: int = 10):
    """
    Create a simple MP4 using a colored background and overlay the title text.
    duration: seconds (should be >= audio duration); we will use audio length by default.
    Requires ffmpeg in PATH.
    """
    # Create a color video of given duration, then combine with audio and overlay text using drawtext.
    # drawtext may require a font path; fallback to default font.
    # We'll attempt to overlay text; if drawtext fails, fallback to simple combine.
    # Use 1280x720 portrait/landscape as needed; we use 1280x720 (16:9).
    # Determine audio duration using ffprobe
    try:
        # get audio duration
        cmd_probe = [FFMPEG_BIN.replace("ffmpeg","ffprobe"), "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", audio_path]
        proc = subprocess.run(cmd_probe, capture_output=True, text=True)
        audio_seconds = float(proc.stdout.strip()) if proc.stdout else duration
    except Exception:
        audio_seconds = duration

    # choose final duration slightly above audio_seconds
    final_dur = max(int(audio_seconds + 0.5), duration)

    tmp_color = output_path + ".color.mp4"
    # create a color video
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-f", "lavfi",
        "-i", f"color=c=0x071013:s=1280x720:d={final_dur}",
        "-vf", f"drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:text='{title}':fontsize=36:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2",
        "-c:v", "libx264",
        "-t", str(final_dur),
        "-pix_fmt", "yuv420p",
        tmp_color
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        # combine with audio
        cmd2 = [
            FFMPEG_BIN,
            "-y",
            "-i", tmp_color,
            "-i", audio_path,
            "-c:v", "libx264",
            "-c:a", "aac",
            "-shortest",
            output_path
        ]
        subprocess.run(cmd2, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        # If drawtext/font problems, fallback to simple combine without text overlay
        try:
            cmd_fallback = [
                FFMPEG_BIN,
                "-y",
                "-f", "lavfi",
                "-i", f"color=c=0x071013:s=1280x720:d={final_dur}",
                "-i", audio_path,
                "-c:v", "libx264",
                "-c:a", "aac",
                "-shortest",
                output_path
            ]
            subprocess.run(cmd_fallback, check=True, capture_output=True)
        except Exception as e2:
            raise RuntimeError(f"ffmpeg failed: {e}\nfallback failed: {e2}")
    finally:
        # cleanup temp
        if os.path.exists(tmp_color):
            try:
                os.remove(tmp_color)
            except: pass

def upload_to_supabase_and_insert(title: str, role: str, file_path: str, quiz: list):
    # create a unique filename
    filename = f"{role.lower()}_{uuid.uuid4().hex}.mp4"
    # upload to bucket
    with open(file_path, "rb") as f:
        res = supabase.storage.from_(SUPABASE_BUCKET).upload(filename, f, {"contentType": "video/mp4"})
    if res.get("error"):
        raise RuntimeError(f"Supabase upload error: {res['error']}")
    # get public url (public bucket) or create signed URL (private)
    # Try public first:
    try:
        pub = supabase.storage.from_(SUPABASE_BUCKET).get_public_url(filename)
        video_url = pub.get("publicURL") or pub.get("data", {}).get("publicUrl") or pub.get("data", {}).get("publicURL")
        if not video_url:
            # try create signed url for 24h
            signed = supabase.storage.from_(SUPABASE_BUCKET).create_signed_url(filename, 60*60*24)
            video_url = signed.get("signedURL") or signed.get("data", {}).get("signedUrl")
    except Exception:
        # fallback
        signed = supabase.storage.from_(SUPABASE_BUCKET).create_signed_url(filename, 60*60*24)
        video_url = signed.get("signedURL") or signed.get("data", {}).get("signedUrl")

    # Insert metadata into videos table
    # Use quiz[0] as primary question for legacy schema; also store JSON options as array
    primary_q = quiz[0] if quiz and len(quiz) > 0 else None
    quiz_question = primary_q.get("question") if primary_q else None
    quiz_options = primary_q.get("options") if primary_q else None
    quiz_answer = primary_q.get("answer") if primary_q else None

    insert_payload = {
        "title": title,
        "video_url": video_url,
        "role": role,
        "quiz_question": quiz_question,
        "quiz_options": json.dumps(quiz_options) if quiz_options else None,
        "quiz_answer": quiz_answer
    }
    # Attempt insert
    resp = supabase.table("videos").insert(insert_payload).execute()
    if resp.get("error"):
        # still return video_url but error on DB insert
        return {"video_url": video_url, "insert_error": resp["error"]}
    return {"video_url": video_url, "db_result": resp.get("data")}

@app.post("/generate")
def generate(req: GenerateRequest):
    # 1) generate script + quiz via HF
    try:
        hf_out = call_hf_generate(req.topic, req.role, req.length_seconds)
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {str(e)}")

    title = hf_out.get("title", f"{req.topic} for {req.role}")
    script = hf_out["script"]
    quiz = hf_out["quiz"]

    # 2) TTS
    tmpdir = tempfile.mkdtemp(prefix="ai_video_")
    audio_path = os.path.join(tmpdir, "audio.mp3")
    try:
        tts_save(script, audio_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")

    # 3) Make video
    video_out = os.path.join(tmpdir, "video.mp4")
    try:
        make_video(audio_path, title, video_out, duration=req.length_seconds)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video creation failed: {str(e)}")

    # 4) Upload and insert
    try:
        up_res = upload_to_supabase_and_insert(title, req.role, video_out, quiz)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload/DB insert failed: {str(e)}")

    # clean temp files (best effort)
    try:
        for f in [audio_path, video_out]:
            if os.path.exists(f):
                os.remove(f)
    except: pass

    return {"ok": True, "title": title, "video_url": up_res.get("video_url"), "quiz": quiz, "meta": up_res.get("db_result") or up_res.get("insert_error")}
