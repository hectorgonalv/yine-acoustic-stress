# scripts/02_build_dictionary.py
import argparse
import os
import re
import sys

# --- Path Hack to allow import from src/ without installing as package ---
# This adds the parent directory (project root) to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.phonetics import get_yine_g2p_rules

def build_dictionary(input_dir, output_file):
    """
    Reads all .txt files in input_dir, tokenizes, applies G2P, and writes dictionary.
    """
    print("Initializing G2P rules...")
    phon_rules, t_s_to_ts, w_to_wɰ = get_yine_g2p_rules()
    
    unique_words = set()
    
    # 1. Gather Words
    files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    print(f"Processing {len(files)} text files from {input_dir}...")

    for filename in files:
        filepath = os.path.join(input_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read().lower()
                # Robust tokenization: replace non-alpha with space
                cleaned_text = re.sub(r'[^a-z\s]', ' ', text)
                tokens = cleaned_text.split()
                unique_words.update(tokens)
        except Exception as e:
            print(f"Warning: Could not read {filename}: {e}")

    sorted_words = sorted(list(unique_words))
    print(f"Found {len(sorted_words)} unique words.")

    # 2. Generate Pronunciations
    print(f"Writing dictionary to {output_file}...")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for word in sorted_words:
            try:
                # Apply main rules
                # Pynini string optimization: (input @ rule).string()
                phon_word = (word @ phon_rules).string()
                
                # Space out phonemes for MFA (e.g. "t s" -> "t s")
                phonemes = ' '.join(list(phon_word))
                
                # Apply cleanup (e.g. reconnect affricates split by spacing)
                phonemes_cleaned = (phonemes @ t_s_to_ts).string()
                
                f.write(f"{word}\t{phonemes_cleaned}\n")

                # Handle "awa" exception (Variant pronunciation)
                if "awa" in word:
                    phon_word_alt = (word @ phon_rules @ w_to_wɰ).string()
                    phonemes_alt = ' '.join(list(phon_word_alt))
                    phonemes_cleaned_alt = (phonemes_alt @ t_s_to_ts).string()
                    f.write(f"{word}\t{phonemes_cleaned_alt}\n")

            except Exception as e:
                # Pynini FstOpError happens if chars are outside the alphabet
                print(f"Skipping word '{word}': Encoding error or invalid character.")

    print("Dictionary generation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Yine pronunciation dictionary from corpus")
    parser.add_argument("--input_dir", required=True, help="Folder containing cleaned .txt files")
    parser.add_argument("--output_file", default="./data/processed/Yine_prondict.txt", help="Output path")
    
    args = parser.parse_args()
    build_dictionary(args.input_dir, args.output_file)