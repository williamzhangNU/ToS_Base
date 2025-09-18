import re
from typing import Tuple
import hashlib
def parse_llm_response(text: str, enable_think: bool = True) -> Tuple[str, str, bool]:
    """Parse LLM response for optional <think> and required <answer> content.

    Returns: (think_content, answer_content, parsed_ok)
    """
    think_pattern = r'<think>(.*?)</think>'
    answer_pattern = r'<answer>(.*?)</answer>'

    think_match = re.search(think_pattern, text, re.DOTALL)
    answer_match = re.search(answer_pattern, text, re.DOTALL)

    think_content = think_match.group(1).strip() if think_match else ""
    answer_content = (answer_match.group(1).strip() if answer_match else text).strip()

    if not enable_think:
        return "", answer_content, bool(answer_content)

    parsed_ok = bool(think_content) and bool(answer_content)
    return think_content, answer_content, parsed_ok

def hash(input_str: str) -> str:
    """Generate a stable hash for the given input string."""
    return hashlib.sha256(input_str.encode('utf-8')).hexdigest()[:16]