import sqlite3
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, Text
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

class LessonRecord(Base):
    __tablename__ = 'lessons'
    id = Column(Integer, primary_key=True, autoincrement=True)
    grade = Column(String(20))
    subject = Column(String(50))
    source_hindi = Column(Text)
    target_language = Column(String(30)) # Santali, Ho, Mundari
    translated_text = Column(Text)
    flashcard_json = Column(Text)
    confidence_score = Column(Float)
    is_human_validated = Column(Boolean, default=False)
    created_at = Column(String(50), default=lambda: datetime.utcnow().isoformat())

class StudentAssessment(Base):
    __tablename__ = 'assessments'
    id = Column(Integer, primary_key=True, autoincrement=True)
    student_name = Column(String(50))
    lesson_id = Column(Integer)
    spoken_response = Column(Text)
    expected_response = Column(Text)
    accuracy_score = Column(Float)
    feedback = Column(Text)

class KalamDatabase:
    def __init__(self, db_url="sqlite:///data/kalam_offline.db"):
        """
        Defaults to SQLite for offline low-cost devices.
        Pass PostgreSQL URI (e.g., postgresql://user:pass@localhost/kalam) for scalable cloud sync.
        """
        self.engine = create_engine(db_url, connect_args={"check_same_thread": False} if "sqlite" in db_url else {})
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def save_lesson(self, grade, subject, hindi_text, target_lang, translated_text, flashcards, confidence):
        session = self.Session()
        lesson = LessonRecord(
            grade=grade, subject=subject, source_hindi=hindi_text,
            target_language=target_lang, translated_text=translated_text,
            flashcard_json=flashcards, confidence_score=confidence
        )
        session.add(lesson)
        session.commit()
        lesson_id = lesson.id
        session.close()
        return lesson_id

    def validate_translation(self, lesson_id, corrected_text):
        """Human-in-the-loop (HITL) validation by native teachers/community"""
        session = self.Session()
        lesson = session.query(LessonRecord).filter_by(id=lesson_id).first()
        if lesson:
            lesson.translated_text = corrected_text
            lesson.is_human_validated = True
            session.commit()
        session.close()

    def save_assessment(self, student_name, lesson_id, spoken, expected, score, feedback):
        session = self.Session()
        assessment = StudentAssessment(
            student_name=student_name, lesson_id=lesson_id,
            spoken_response=spoken, expected_response=expected,
            accuracy_score=score, feedback=feedback
        )
        session.add(assessment)
        session.commit()
        session.close()

    def get_all_lessons(self):
        session = self.Session()
        lessons = session.query(LessonRecord).order_by(LessonRecord.id.desc()).all()
        session.close()
        return lessons