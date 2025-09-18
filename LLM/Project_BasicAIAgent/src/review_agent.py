# src/review_agent.py
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load a small open-source model for speed (can swap with GPT-3.5 later)
MODEL_NAME = "google/flan-t5-base"  # Example small open LLM
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)


def review_doc(text: str) -> str:
    """
    Returns review feedback for a documentation snippet.
    """
    prompt = (
    """You are an expert technical documentation reviewer. 
    For each paragraph, provide grammar mistakes

    Example:
    Input: "The cat are on the roof."
    Output: Grammar mistake, it should be The cats are on the roof

    Now review this text: {text}"""
    )
    output = generator(prompt, max_new_tokens=500,  do_sample=False, early_stopping=True)[0]["generated_text"]
    # Remove input text from output if included
    return output
