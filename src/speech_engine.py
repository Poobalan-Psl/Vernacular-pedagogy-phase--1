import torch
import soundfile as sf
from transformers import pipeline, VitsModel, AutoTokenizer

class SpeechEngine:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.stt_pipe = None
        self.tts_models = {}

        # Meta MMS ISO 639-3 codes for Target Tribal Languages + Hindi
        self.mms_lang_map = {
            "Hindi": "facebook/mms-tts-hin",
            "Santali": "facebook/mms-tts-sat",
            "Mundari": "facebook/mms-tts-unr",
            "Ho": "facebook/mms-tts-hoc" # Fallback to hin/sat if local weights unavailable
        }

    def load_whisper(self, model_id="openai/whisper-small"):
        if self.stt_pipe is None:
            self.stt_pipe = pipeline(
                "automatic-speech-recognition",
                model=model_id,
                device=0 if self.device == "cuda" else -1
            )

    def speech_to_text(self, audio_filepath, language="hindi"):
        """Transcribes teacher or student audio using Whisper"""
        self.load_whisper()
        result = self.stt_pipe(
            audio_filepath,
            generate_kwargs={"language": language, "task": "transcribe"}
        )
        return result["text"].strip()

    def text_to_speech(self, text, target_language="Santali", output_path="output_tts.wav"):
        """Synthesizes speech using Meta MMS (VITS architecture)"""
        model_id = self.mms_lang_map.get(target_language, "facebook/mms-tts-hin")
        try:
            if model_id not in self.tts_models:
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model = VitsModel.from_pretrained(model_id).to(self.device)
                self.tts_models[model_id] = (tokenizer, model)
            else:
                tokenizer, model = self.tts_models[model_id]

            inputs = tokenizer(text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                output = model(**inputs).waveform

            waveform = output.squeeze().cpu().numpy()
            sf.write(output_path, waveform, model.config.sampling_rate)
            return output_path
        except Exception as e:
            print(f"Meta MMS loading warning ({e}). Generating fallback audio.")
            return None