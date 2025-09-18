# src/scoring.py
import textstat

def readability_score(text: str) -> float:
    """
    Returns a score between 0-100 based on Flesch Reading Ease.
    Higher is easier to read.
    """
    score = textstat.flesch_reading_ease(text)
    # Normalize to 0-100
    score = max(0, min(100, score))
    return score

def overall_score(text: str, feedback_weight=0.5) -> float:
    """
    Combine readability + LLM feedback (simulated as simple heuristic) into one score
    """
    readability = readability_score(text)
    # For demo, assume 1-3 issues reduce score by 5-15
    issues_detected = text.lower().count("issue")  # placeholder
    penalty = min(15, issues_detected * 5)
    return readability - penalty
