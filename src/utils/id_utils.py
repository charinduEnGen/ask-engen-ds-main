import uuid
from datetime import datetime, timezone


def generate_answer_id() -> str:
    """e.g. ans_20260724_162200_4f2a9c — matches the "ans_<ts>_<seq>" contract."""
    now = datetime.now(timezone.utc)
    return f"ans_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


def generate_question_id() -> str:
    """e.g. qst_20260724_162215_4f2a9c — server-assigned, echoed back as questionId."""
    now = datetime.now(timezone.utc)
    return f"qst_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
