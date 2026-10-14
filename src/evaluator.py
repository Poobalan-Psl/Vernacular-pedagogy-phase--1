import sacrebleu
from jiwer import wer

class KalamEvaluator:
    @staticmethod
    def evaluate_translation(predicted_text: str, reference_text: str):
        """Computes BLEU and chrF scores for Hindi -> Tribal Translation"""
        bleu = sacrebleu.corpus_bleu([predicted_text], [[reference_text]])
        chrf = sacrebleu.corpus_chrf([predicted_text], [[reference_text]])
        return {
            "BLEU": round(bleu.score, 2),
            "chrF": round(chrf.score, 2)
        }

    @staticmethod
    def evaluate_speech_wer(transcribed_speech: str, ground_truth_speech: str):
        """
        Computes Word Error Rate (WER) for ASR accuracy and Student Practice Feedback.
        """
        if not ground_truth_speech.strip():
            return {"WER": 0.0, "Accuracy": 100.0}
        error_rate = wer(ground_truth_speech.lower(), transcribed_speech.lower())
        accuracy = max(0.0, round((1.0 - error_rate) * 100, 2))
        return {
            "WER": round(error_rate * 100, 2),
            "Accuracy_Percent": accuracy
        }