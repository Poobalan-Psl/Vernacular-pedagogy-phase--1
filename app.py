import os
import json
import streamlit as st
from src.database import KalamDatabase
from src.ocr_pipeline import TextbookOCR
from src.speech_engine import SpeechEngine
from src.translation_pedagogy import VernacularPedagogyEngine
from src.evaluator import KalamEvaluator

st.set_page_config(page_title="KALAM - SIH26042 (ZERO PIXEL)", layout="wide")

# Initialize Core Singletons
@st.cache_resource
def init_system():
    os.makedirs("data", exist_ok=True)
    db = KalamDatabase("sqlite:///data/kalam_offline.db")
    ocr = TextbookOCR()
    speech = SpeechEngine()
    pedagogy = VernacularPedagogyEngine(lite_mode=True) # Lite Mode ON by default for fast offline run
    evaluator = KalamEvaluator()
    return db, ocr, speech, pedagogy, evaluator

db, ocr, speech, pedagogy, evaluator = init_system()

st.title("🎓 KALAM: Knowledge Assistance for Language-based Accessible Multilingual Learning")
st.caption("SIH26042 | Team ZERO PIXEL | Offline-First Vernacular Pedagogy & Real-Time Translation (Santali, Ho, Mundari)")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Device & Classroom Settings")
    target_lang = st.selectbox("Select Student Mother Tongue", ["Santali", "Mundari", "Ho"])
    grade_level = st.selectbox("Select Grade", ["Grade 1", "Grade 2", "Grade 3", "Grade 4", "Grade 5"])
    subject = st.selectbox("Subject", ["Language (FLN)", "Mathematics", "Environmental Studies (EVS)"])
    lite_mode = st.toggle("Offline Lite Mode (Quantized / Low-Memory)", value=True)
    pedagogy.lite_mode = lite_mode
    st.info("Offline SQLite Storage: Active (`data/kalam_offline.db`)")

tab1, tab2, tab3, tab4 = st.tabs([
    "🗣️ 1. Teacher Input & Translation",
    "🃏 2. Flashcards & Interactive Learning Loop",
    "✅ 3. Teacher / Community Validation (HITL)",
    "📊 4. AI Evaluation (BLEU, chrF, WER)"
])

# TAB 1: TEACHER INPUT (TEXT / VOICE / OCR) -> VERNACULAR TRANSLATION
with tab1:
    st.subheader("Teacher-to-Student Vernacular Bridge")
    input_mode = st.radio("Select Input Mode", ["Direct Hindi Text / Quick Phrase", "Textbook OCR Scanner (OpenCV)", "Real-Time Voice Recording (Whisper STT)"], horizontal=True)

    hindi_input = ""
    if input_mode == "Direct Hindi Text / Quick Phrase":
        sample_phrases = [
            "नमस्ते बच्चों",
            "आज हम गणित पढ़ेंगे",
            "पेड़ हमें फल और हवा देते हैं",
            "पानी जीवन है",
            "Custom Input..."
        ]
        choice = st.selectbox("Choose Classroom Phrase or Type Custom:", sample_phrases)
        hindi_input = st.text_area("Hindi Lesson Input:", value="" if choice == "Custom Input..." else choice)

    elif input_mode == "Textbook OCR Scanner (OpenCV)":
        uploaded_img = st.file_uploader("Upload Textbook Page Image", type=["png", "jpg", "jpeg"])
        if uploaded_img:
            extracted = ocr.extract_text(uploaded_img.read())
            hindi_input = st.text_area("Extracted OCR Text (Editable):", value=extracted)

    elif input_mode == "Real-Time Voice Recording (Whisper STT)":
        audio_file = st.file_uploader("Upload Teacher Audio (.wav / .mp3)", type=["wav", "mp3"])
        if audio_file and st.button("Transcribe Teacher Voice"):
            with open("data/temp_teacher.wav", "wb") as f:
                f.write(audio_file.read())
            hindi_input = speech.speech_to_text("data/temp_teacher.wav", language="hindi")
            st.success(f"Whisper Transcription: {hindi_input}")

    if st.button("🚀 Translate & Generate Lesson Material", type="primary"):
        if hindi_input.strip():
            translated_text, confidence = pedagogy.translate_with_confidence(hindi_input, target_lang)
            flashcards_json = pedagogy.generate_pedagogy_materials(hindi_input, translated_text, target_lang, grade_level)
            
            lesson_id = db.save_lesson(
                grade=grade_level, subject=subject, hindi_text=hindi_input,
                target_lang=target_lang, translated_text=translated_text,
                flashcards=flashcards_json, confidence=confidence
            )
            st.session_state["current_lesson"] = {
                "id": lesson_id,
                "hindi": hindi_input,
                "translated": translated_text,
                "lang": target_lang,
                "confidence": confidence,
                "flashcards": json.loads(flashcards_json)
            }
            st.success(f"Translated to {target_lang} (Confidence Score: {confidence*100:.0f}%)")
            st.markdown(f"### 🗣️ **{target_lang} Output:** `{translated_text}`")

            # Generate TTS Audio via Meta MMS
            audio_out = speech.text_to_speech(translated_text, target_language=target_lang, output_path="data/lesson_tts.wav")
            if audio_out and os.path.exists(audio_out):
                st.audio(audio_out, format="audio/wav")

# TAB 2: INTERACTIVE LEARNING LOOP (LISTEN -> PRACTICE -> FEEDBACK -> IMPROVE)
with tab2:
    st.subheader("🔄 Interactive Student Learning Loop")
    if "current_lesson" not in st.session_state:
        st.warning("Please generate a lesson in Tab 1 first.")
    else:
        lesson = st.session_state["current_lesson"]
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 1️⃣ Listen & Learn (Visual Flashcards)")
            for card in lesson["flashcards"].get("flashcards", []):
                with st.container(border=True):
                    st.markdown(f"## {card['visual_icon']}")
                    st.write(f"Hindi:")
                    st.write(f"**{lesson['lang']}:** `{card['vernacular_term']}`")
                    st.caption(card["pronunciation_guide"])

        with col2:
            st.markdown("#### 2️⃣ Practice, Get Feedback & Improve")
            student_name = st.text_input("Student Name", value="Birsa")
            student_spoken = st.text_input(
                f"Simulate Student Speaking / Typing in {lesson['lang']}:",
                value=lesson["translated"]
            )
            if st.button("🎯 Evaluate Student Response"):
                metrics = evaluator.evaluate_speech_wer(student_spoken, lesson["translated"])
                acc = metrics["Accuracy_Percent"]
                feedback = "🌟 Excellent pronunciation and comprehension!" if acc >= 80 else "🌱 Good effort! Listen to the audio flashcard and try once more."
                
                db.save_assessment(student_name, lesson["id"], student_spoken, lesson["translated"], acc, feedback)
                st.metric("Student Comprehension & Speech Score", f"{acc}%", f"WER: {metrics['WER']}%")
                st.info(feedback)

# TAB 3: HUMAN-IN-THE-LOOP (HITL) TEACHER & COMMUNITY VALIDATION
with tab3:
    st.subheader("👩‍🏫 Human-in-the-Loop (HITL) Community Validation")
    st.caption("Enables teachers and native speakers to validate Ho, Mundari, and Santali content for cultural and educational accuracy.")
    lessons = db.get_all_lessons()
    for l in lessons:
        with st.expander(f"Lesson #{l.id} | {l.grade} - {l.subject} | {l.target_language} (Confidence: {l.confidence_score})"):
            st.write(f"**Source Hindi:** {l.source_hindi}")
            new_trans = st.text_input(f"Validated {l.target_language} Translation:", value=l.translated_text, key=f"val_{l.id}")
            status = "✅ Validated by Community" if l.is_human_validated else "⏳ Pending Human Review"
            st.write(f"**Status:** {status}")
            if st.button("Approve & Update Local Dataset", key=f"btn_{l.id}"):
                db.validate_translation(l.id, new_trans)
                st.success("Updated in Offline SQLite DB for continuous fine-tuning!")

# TAB 4: EVALUATION METRICS (BLEU, chrF, WER)
with tab4:
    st.subheader("📈 Benchmark Evaluation Suite (BLEU, chrF & WER)")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Translation Quality (BLEU & chrF)")
        hyp = st.text_input("Model Translation (Hypothesis)", "ᱫᱟᱜ ᱜᱮ ᱡᱤᱭᱚᱱ ᱠᱟᱱᱟ")
        ref = st.text_input("Community Gold Reference", "ᱫᱟᱜ ᱜᱮ ᱡᱤᱭᱚᱱ ᱠᱟᱱᱟ")
        if st.button("Compute BLEU & chrF"):
            res = evaluator.evaluate_translation(hyp, ref)
            st.json(res)
    with c2:
        st.markdown("##### Speech Recognition Accuracy (WER)")
        asr_out = st.text_input("Whisper STT Output", "आज हम गणित पढ़ेंगे")
        asr_ref = st.text_input("Ground Truth Audio Transcript", "आज हम गणित पढ़ेंगे")
        if st.button("Compute WER"):
            res_wer = evaluator.evaluate_speech_wer(asr_out, asr_ref)
            st.json(res_wer)