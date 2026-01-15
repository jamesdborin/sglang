#!/usr/bin/env python3
"""
Simple test script for NGRAMGUESS feature.

This tests the parsing function directly without importing sglang modules.
"""

import re
from typing import Optional, Tuple


def parse_ngram_guess(text: str) -> Tuple[str, Optional[str]]:
    """
    Parse <NGRAMGUESS> tags from input text for n-gram speculation.
    
    Args:
        text: Input text possibly containing <NGRAMGUESS>...</NGRAMGUESS>
        
    Returns:
        (prompt, guess) tuple where:
        - prompt: The cleaned input text with tags removed
        - guess: The extracted guess text, or None if no tags found
    """
    pattern = r'<NGRAMGUESS>(.*?)</NGRAMGUESS>'
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        guess = match.group(1)
        # Remove the tag and its contents from the original text
        prompt = text[:match.start()] + text[match.end():]
        return prompt, guess
    
    return text, None


def test_parse_ngram_guess():
    """Test the parse_ngram_guess function."""
    print("Testing parse_ngram_guess()...")
    
    # Test 1: Simple case
    text1 = "The capital of France is <NGRAMGUESS>Paris</NGRAMGUESS>"
    prompt1, guess1 = parse_ngram_guess(text1)
    assert prompt1 == "The capital of France is ", f"Expected 'The capital of France is ', got '{prompt1}'"
    assert guess1 == "Paris", f"Expected 'Paris', got '{guess1}'"
    print("✓ Test 1 passed: Simple case")
    
    # Test 2: No tags
    text2 = "Hello world"
    prompt2, guess2 = parse_ngram_guess(text2)
    assert prompt2 == "Hello world", f"Expected 'Hello world', got '{prompt2}'"
    assert guess2 is None, f"Expected None, got '{guess2}'"
    print("✓ Test 2 passed: No tags")
    
    # Test 3: Empty guess
    text3 = "Test<NGRAMGUESS></NGRAMGUESS>"
    prompt3, guess3 = parse_ngram_guess(text3)
    assert prompt3 == "Test", f"Expected 'Test', got '{prompt3}'"
    assert guess3 == "", f"Expected '', got '{guess3}'"
    print("✓ Test 3 passed: Empty guess")
    
    # Test 4: Multiline guess
    text4 = "Question<NGRAMGUESS>Answer\nwith multiple\nlines</NGRAMGUESS>"
    prompt4, guess4 = parse_ngram_guess(text4)
    assert prompt4 == "Question", f"Expected 'Question', got '{prompt4}'"
    assert guess4 == "Answer\nwith multiple\nlines", f"Expected multiline, got '{guess4}'"
    print("✓ Test 4 passed: Multiline guess")
    
    # Test 5: Guess in middle
    text5 = "Start<NGRAMGUESS>middle</NGRAMGUESS>End"
    prompt5, guess5 = parse_ngram_guess(text5)
    assert prompt5 == "StartEnd", f"Expected 'StartEnd', got '{prompt5}'"
    assert guess5 == "middle", f"Expected 'middle', got '{guess5}'"
    print("✓ Test 5 passed: Guess in middle")
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_parse_ngram_guess()
