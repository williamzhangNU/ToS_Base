import re
from typing import Tuple

def extract_think_and_answer(text: str) -> Tuple[str, str]:
    """Extract think and answer content from text using regex patterns"""
    think_pattern = r'<think>(.*?)</think>'
    answer_pattern = r'<answer>(.*?)</answer>'
    
    think_match = re.search(think_pattern, text, re.DOTALL)
    answer_match = re.search(answer_pattern, text, re.DOTALL)
    
    think_content = think_match.group(1).strip() if think_match else ""
    answer_content = answer_match.group(1).strip() if answer_match else text
    
    return think_content, answer_content