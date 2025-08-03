import whisper
import os
import subprocess
import warnings

# Suppress FP16 CPU warning
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# Paths
ORIGINAL_AUDIO = "audio/malayalam.mp3"
WAV_AUDIO = "audio/clean.wav"
OUTPUT_PATH = "transcripts/malayalam.txt"

def convert_to_wav(input_path, output_path):
    print("🔁 Converting to WAV format...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        output_path
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("✅ Conversion complete.")
    except subprocess.CalledProcessError as e:
        print(f"❌ ffmpeg conversion failed: {e}")
        exit(1)

def detect_language(audio_path, model):
    print("🌍 Detecting language...")
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    _, probs = model.detect_language(mel)
    detected = max(probs, key=probs.get)
    print(f"🗣 Detected Language: {detected} ({round(probs[detected]*100, 2)}%)")
    return detected

def transcribe(audio_path, output_path):
    print("🔄 Loading Whisper model...")
    model = whisper.load_model("medium")  # Try "large" if accuracy is poor

    # Detect actual language first
    detected_lang = detect_language(audio_path, model)
    if detected_lang != "ml":
        print("⚠️ Detected language is not Malayalam. Proceeding anyway...")

    print(f"▶️ Transcribing: {os.path.basename(audio_path)}")

    try:
        result = model.transcribe(
            audio_path,
            task="transcribe",
            language="ml",  # Force Malayalam
            verbose=True,
            condition_on_previous_text=False,
            initial_prompt="ഇത് മലയാളം ഭാഷയിലാണ്."  # Helps context
        )

        segments = result.get("segments", [])
        full_text = ""

        if not segments:
            print("⚠️ No text was transcribed.")
        else:
            for i, seg in enumerate(segments):
                print(f"⏱️ {seg['start']:.2f}–{seg['end']:.2f}s: {seg['text']}")
                full_text += seg["text"].strip() + "\n"

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(full_text)

            print(f"\n✅ Transcript saved to: {output_path}")
    except Exception as e:
        print(f"❌ An error occurred during transcription: {e}")

if __name__ == "__main__":
    if not os.path.exists(ORIGINAL_AUDIO):
        print(f"❌ File not found: {ORIGINAL_AUDIO}")
        exit()

    convert_to_wav(ORIGINAL_AUDIO, WAV_AUDIO)
    transcribe(WAV_AUDIO, OUTPUT_PATH)
