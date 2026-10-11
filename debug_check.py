"""Test the fixed search logic directly."""
import sys
import os
sys.path.insert(0, ".")
os.environ["PYTHONIOENCODING"] = "utf-8"

# Import only the helper functions from the server (not the full MCP app)
import re

def _tokenize_filename(name):
    stem = os.path.splitext(name)[0]
    stem = re.sub(r'[_\-\.\s]+', ' ', stem)
    return set(stem.lower().split())

def _keyword_matches_file(keywords, filename):
    tokens = _tokenize_filename(filename)
    raw = filename.lower()
    score = 0
    for kw in keywords:
        token_hit = kw in tokens
        boundary_hit = bool(re.search(r'(?<![a-z0-9])' + re.escape(kw) + r'(?![a-z0-9])', raw))
        if not (token_hit or boundary_hit):
            return 0
        score += 1
    return score

# Test cases
keywords_pan_card = ["pan", "card"]
test_files = [
    # Should MATCH
    ("Pan Card.jpg", True),
    ("pan card scan.pdf", True),
    ("my_pan_card.pdf", True),
    ("PAN_CARD_2024.jpg", True),
    ("Mohith PAN.pdf", False),   # has "pan" but not "card"
    # Should NOT match
    ("companion.msix", False),
    ("lifespan.py", False),
    ("pandas1_practise.ipynb", False),
    ("Mohith ID Card.jpg", False),   # has "card" but not "pan"
    ("saveetha id card.jpg", False),  # same
    ("English Answer Sheet.pdf", False),
]

print("=== Testing 'pan card' search ===")
print(f"Keywords: {keywords_pan_card}")
print()
all_passed = True
for filename, expected in test_files:
    score = _keyword_matches_file(keywords_pan_card, filename)
    matched = score > 0
    status = "PASS" if matched == expected else "FAIL"
    if status == "FAIL":
        all_passed = False
    exp_str = "match" if expected else "no match"
    got_str = f"score={score}" if matched else "no match"
    print(f"  [{status}] '{filename}' -> expected {exp_str}, got {got_str}")

print()
if all_passed:
    print("All tests PASSED!")
else:
    print("Some tests FAILED - check logic")

print()
# Test with "id card"
keywords_id = ["id", "card"]
print("=== Testing 'id card' search ===")
id_tests = [
    ("Mohith ID Card.jpg", True),
    ("saveetha id card.jpg", True),
    ("Pan Card.jpg", False),
    ("companion.msix", False),
]
for filename, expected in id_tests:
    score = _keyword_matches_file(keywords_id, filename)
    matched = score > 0
    status = "PASS" if matched == expected else "FAIL"
    print(f"  [{status}] '{filename}' -> {'match' if matched else 'no match'} (expected {'match' if expected else 'no match'})")

print()
# Now run the actual search
print("=== Running actual search_local_files('pan card') ===")
import importlib
import database as db
db.initialize_database()

from pathlib import Path
LOCAL_SEARCH_DIRS = [
    str(Path.home() / "Documents"),
    str(Path.home() / "Downloads"),
    str(Path.home() / "Desktop"),
]

_SKIP_DIRS = {
    'venv', '.venv', 'node_modules', '__pycache__', '.git', '.tox',
    'site-packages', 'dist-packages', 'dist', 'build', '.mypy_cache',
    '.pytest_cache', 'env', 'Lib', 'lib',
}

query = "pan card"
stop_words = {'the', 'of', 'a', 'an', 'and', 'or', 'in', 'on', 'at', 'to',
              'for', 'with', 'by', 'from', 'file', 'show', 'me', 'get', 'find',
              'my', 'search', 'can', 'you'}
keywords = [w.lower() for w in query.split() if w.lower() not in stop_words and len(w) > 1]
print(f"Keywords: {keywords}")

results = []
from datetime import datetime
for directory in LOCAL_SEARCH_DIRS:
    if not os.path.exists(directory):
        print(f"  {directory}: does not exist")
        continue
    try:
        for root, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in _SKIP_DIRS]
            for f in files:
                score = _keyword_matches_file(keywords, f)
                if score > 0:
                    results.append((score, f, os.path.join(root, f)))
    except (PermissionError, OSError):
        pass

results.sort(reverse=True)
print(f"\nFound {len(results)} matching files:")
for score, name, path in results[:10]:
    print(f"  [score={score}] {name}")
    print(f"    -> {path}")
