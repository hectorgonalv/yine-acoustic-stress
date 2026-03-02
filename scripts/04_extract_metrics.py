# scripts/04_extract_metrics.py
import argparse
import os
import re
import numpy as np
import pandas as pd
import parselmouth
import textgrid
from textgrid import TextGrid, IntervalTier
import sys

# --- Path Hack to allow import from src/ ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.syllabification import split_yine_syllables, syllabify_yine_phones, YINE_VOWELS

# ==============================================================================
# CONFIGURATION
# ==============================================================================
HEADER_ROW =[
    'File_Name', 'Speaker', 'Utterance_id', 'Words', 'Word_Syllable_Count',
    'Syllable', 'Syllable_Phones', 'Syllable_Index', 'Syllable_position',
    'Word_Index_in_Utt', 'Word_Index_Reversed', 'Phrase_Position',
    'Is_Interrogative', 'Is_Exclamative',
    'Preceded_by_Silence', 'Followed_by_Silence',
    'Syllable_Start', 'Syllable_End', 'Syllable_Duration',
    'Vowel', 'Start_Time', 'Quartile_2', 'End_Time', 'Duration',
    'Intensity_Q2', 'Pitch_Q2', 'F1_mid', 'F2_mid', 'Spectral_Tilt_DB',
    'Utterance_Start', 'Utterance_End', 'Utterance_Duration', 'Utt_Syll_Count', 'Pace_SPS'
]

# ==============================================================================
# HELPER FUNCTIONS (From your original pipeline)
# ==============================================================================
def load_source_files(textgrid_path, audio_path):
    try:
        tg = textgrid.TextGrid.fromFile(textgrid_path)
        sound = parselmouth.Sound(audio_path)
        intensity = sound.to_intensity()
        pitch = sound.to_pitch()
        return tg, sound, intensity, pitch
    except Exception as e:
        print(f"  -> Error loading files: {e}")
        return None, None, None, None

def get_word_contexts_from_text(text_file_path):
    try:
        with open(text_file_path, 'r', encoding='utf-8') as f:
            full_text = f.read()
    except FileNotFoundError:
        return[]

    word_contexts =[]
    utterance_counter = 0
    base_filename = os.path.splitext(os.path.basename(text_file_path))[0]

    delimiter_pattern = r'([.?!,;:“”"()—-]+|[¿¡])'
    split_list = re.split(delimiter_pattern, full_text)

    for i in range(0, len(split_list), 2):
        utterance_text = split_list[i]
        following_delimiter = split_list[i + 1] if i + 1 < len(split_list) else ""
        cleaned_utterance = utterance_text.strip()
        
        if not cleaned_utterance:
            continue

        is_interrogative = cleaned_utterance.startswith('¿') or '?' in following_delimiter
        is_exclamative = cleaned_utterance.startswith('¡') or '!' in following_delimiter
        utterance_id = f"{base_filename}_utt_{utterance_counter}"

        lower_text = cleaned_utterance.lower()
        cleaned_text_for_words = re.sub(r'[^a-z\s]', ' ', lower_text)
        words = cleaned_text_for_words.split()

        if words:
            for word in words:
                word_contexts.append((word, utterance_id, is_interrogative, is_exclamative))
            utterance_counter += 1

    return word_contexts

def calculate_positional_metrics(word_contexts):
    if not word_contexts: return[]

    grouped_by_utterance = {}
    for word, utt_id, is_interrog, is_exclam in word_contexts:
        if utt_id not in grouped_by_utterance:
            grouped_by_utterance[utt_id] = []
        grouped_by_utterance[utt_id].append((word, utt_id, is_interrog, is_exclam))

    contexts_with_position =[]
    for utterance_id, words_in_utterance in grouped_by_utterance.items():
        num_words = len(words_in_utterance)
        for i, (word, utt_id, is_interrog, is_exclam) in enumerate(words_in_utterance):
            phrase_position = 'final' if i == num_words - 1 else \
                              'midfinal' if i == num_words - 2 else \
                              'initial' if i == 0 else 'medial'
            contexts_with_position.append((
                word, utt_id, is_interrog, is_exclam, i, num_words - 1 - i, phrase_position
            ))

    return contexts_with_position

def get_good_word_intervals_from_mfa(mfa_textgrid):
    word_tier = mfa_textgrid.tiers[0]
    phone_tier = mfa_textgrid.tiers[1]
    good_word_intervals =[]

    for word_interval in word_tier:
        word_mark = word_interval.mark.strip()
        if not word_mark or (word_mark.startswith('[') and word_mark.endswith(']')):
            continue
        
        phones_in_interval =[p for p in phone_tier if p.minTime >= word_interval.minTime and p.maxTime <= word_interval.maxTime and p.mark]
        if not phones_in_interval: continue
        
        phone_marks = [p.mark.strip() for p in phones_in_interval]
        if len(phone_marks) == 1 and phone_marks[0] == 'spn': continue
        
        good_word_intervals.append(word_interval)
    return good_word_intervals

def reconcile_word_lists(ortho_word_contexts, mfa_good_word_intervals):
    synchronized_ortho_contexts = []
    synchronized_intervals =[]
    ortho_idx, mfa_idx = 0, 0

    while ortho_idx < len(ortho_word_contexts):
        if mfa_idx >= len(mfa_good_word_intervals):
            ortho_idx += 1
            continue

        ortho_word = ortho_word_contexts[ortho_idx][0]
        mfa_word = mfa_good_word_intervals[mfa_idx].mark.strip().casefold()

        if ortho_word == mfa_word:
            synchronized_ortho_contexts.append(ortho_word_contexts[ortho_idx])
            synchronized_intervals.append(mfa_good_word_intervals[mfa_idx])
            ortho_idx += 1
            mfa_idx += 1
        else:
            ortho_idx += 1

    return synchronized_ortho_contexts, synchronized_intervals

def get_vowel_metrics(tg, speaker_tg, synchronized_contexts, synchronized_intervals, sound, intensity, pitch, base_filename):
    speaker_tier = speaker_tg.tiers[0]
    text_entries =[]
    
    for word, utt_id, is_interrog, is_exclam, word_idx, rev_idx, phrase_pos in synchronized_contexts:
        syllables = split_yine_syllables(word)
        num_syllables = len(syllables)
        for k, syllable in enumerate(syllables):
            text_entries.append([
                word, syllable, num_syllables - 1 - k, None, utt_id,
                is_interrog, is_exclam, word_idx, rev_idx, phrase_pos
            ])

    temp_tg = TextGrid(minTime=tg.minTime, maxTime=tg.maxTime)
    clean_word_tier = IntervalTier(name="words", minTime=tg.minTime, maxTime=tg.maxTime)
    for interval in synchronized_intervals:
        clean_word_tier.add(interval.minTime, interval.maxTime, interval.mark)
    temp_tg.append(clean_word_tier)
    temp_tg.append(tg.tiers[1])
    
    phonetic_syllables = syllabify_yine_phones(temp_tg)

    if len(text_entries) != len(phonetic_syllables):
        return[]

    formants = sound.to_formant_burg(time_step=0.01, max_number_of_formants=5, maximum_formant=5500, window_length=0.025)
    final_data_rows =[]

    for i in range(len(text_entries)):
        word, syllable_str, syl_pos, _, utt_id, is_interrog, is_exclam, word_idx, rev_idx, phrase_pos = text_entries[i]
        syllable_phones = phonetic_syllables[i]

        vowel_phones = [p for p in syllable_phones if p.mark in YINE_VOWELS]
        if not vowel_phones: continue

        vowel_start_time = vowel_phones[0].minTime
        vowel_end_time = vowel_phones[-1].maxTime
        vowel_midpoint = vowel_start_time + ((vowel_end_time - vowel_start_time) / 2)

        speaker_id = "silence"
        for interval in speaker_tier:
            if interval.minTime <= vowel_midpoint < interval.maxTime:
                speaker_id = interval.mark
                break

        syllable_start_time = syllable_phones[0].minTime
        syllable_end_time = syllable_phones[-1].maxTime
        syllable_phones_str = " ".join([p.mark for p in syllable_phones])

        vowel_dur = vowel_end_time - vowel_start_time
        f1_mid = formants.get_value_at_time(1, vowel_midpoint)
        f2_mid = formants.get_value_at_time(2, vowel_midpoint)
        f1_mid = 0 if np.isnan(f1_mid) else f1_mid
        f2_mid = 0 if np.isnan(f2_mid) else f2_mid

        spectral_tilt = 0.0
        try:
            sound_slice = sound.extract_part(from_time=vowel_midpoint - 0.015, to_time=vowel_midpoint + 0.015, preserve_times=True)
            spectral_tilt = sound_slice.to_spectrum().get_band_energy_difference(0, 1000, 1000, 4000)
            if np.isnan(spectral_tilt): spectral_tilt = 0.0
        except: pass

        intensity_q2 = intensity.get_value(vowel_midpoint)
        pitch_q2 = pitch.get_value_at_time(vowel_midpoint)
        pitch_q2 = 0 if np.isnan(pitch_q2) else pitch_q2

        final_data_rows.append([
            base_filename, speaker_id, utt_id, word, len(split_yine_syllables(word)),
            syllable_str, syllable_phones_str, len(split_yine_syllables(word)) - 1 - syl_pos, syl_pos,
            word_idx, rev_idx, phrase_pos, is_interrog, is_exclam, None, None,
            syllable_start_time, syllable_end_time, syllable_end_time - syllable_start_time,
            vowel_phones[0].mark, vowel_start_time, vowel_midpoint, vowel_end_time, vowel_dur,
            intensity_q2, pitch_q2, f1_mid, f2_mid, spectral_tilt
        ])

    return final_data_rows

def add_breath_group_metrics(vowels_data, tg):
    phone_tier = tg.tiers[1]
    silence_starts, silence_ends = set(), set()
    for interval in phone_tier:
        if (not interval.mark or not interval.mark.strip()) and (interval.maxTime - interval.minTime > 0.1):
            silence_starts.add(round(interval.minTime, 4))
            silence_ends.add(round(interval.maxTime, 4))

    for row in vowels_data:
        row[14] = round(row[16], 4) in silence_ends
        row[15] = round(row[17], 4) in silence_starts
    return vowels_data

def calculate_and_add_pace(vowels_data):
    grouped_by_utterance = {}
    for row in vowels_data:
        grouped_by_utterance.setdefault(row[2],[]).append(row)

    metrics_map = {}
    for utt_id, rows in grouped_by_utterance.items():
        start_time, end_time = rows[0][16], rows[-1][17]
        duration = end_time - start_time
        metrics_map[utt_id] = {
            'start': start_time, 'end': end_time, 'duration': duration,
            'count': len(rows), 'pace': (len(rows) / duration) if duration > 0 else 0.0
        }

    return [row + [metrics_map[row[2]]['start'], metrics_map[row[2]]['end'], metrics_map[row[2]]['duration'], metrics_map[row[2]]['count'], metrics_map[row[2]]['pace']] for row in vowels_data]

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def process_corpus(txt_dir, wav_dir, mfa_dir, diarize_dir, output_csv):
    all_files_data = []
    
    txt_files =[f for f in os.listdir(txt_dir) if f.endswith('.txt')]
    print(f"Found {len(txt_files)} text files.")

    for filename in txt_files:
        base = os.path.splitext(filename)[0]
        paths = {
            'txt': os.path.join(txt_dir, f"{base}.txt"),
            'wav': os.path.join(wav_dir, f"{base}.wav"),
            'mfa': os.path.join(mfa_dir, f"{base}.TextGrid"),
            'diarize': os.path.join(diarize_dir, f"{base}_speaker.TextGrid")
        }

        # Check if ALL files exist (including MFA, which you will generate manually later)
        if not all(os.path.exists(p) for p in paths.values()):
            print(f"Skipping {base}: Missing associated .wav or .TextGrid files.")
            continue

        print(f"Processing: {base}...")
        tg, sound, intensity, pitch = load_source_files(paths['mfa'], paths['wav'])
        speaker_tg = textgrid.TextGrid.fromFile(paths['diarize'])

        ortho_contexts = get_word_contexts_from_text(paths['txt'])
        contexts_w_pos = calculate_positional_metrics(ortho_contexts)
        mfa_good = get_good_word_intervals_from_mfa(tg)
        final_ortho, final_intervals = reconcile_word_lists(contexts_w_pos, mfa_good)

        if final_ortho:
            measurements = get_vowel_metrics(tg, speaker_tg, final_ortho, final_intervals, sound, intensity, pitch, base)
            if measurements:
                data_breath = add_breath_group_metrics(measurements, tg)
                final_data = calculate_and_add_pace(data_breath)
                all_files_data.extend(final_data)

    if all_files_data:
        print("\nCreating final DataFrame...")
        df = pd.DataFrame(all_files_data, columns=HEADER_ROW)
        
        # Sanitize data for CSV
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        
        # Round numerics
        numeric_cols = df.select_dtypes(include=np.number).columns
        df[numeric_cols] = df[numeric_cols].round(4)

        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"[V] Successfully saved {len(df)} extracted metrics to {output_csv}")
    else:
        print("\n[!] No complete data sets found. Did you generate the MFA TextGrids yet?")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Acoustic Metrics")
    parser.add_argument("--txt_dir", default="./data/raw")
    parser.add_argument("--wav_dir", default="./data/raw")
    parser.add_argument("--mfa_dir", default="./data/mfa_alignments", help="Folder with MFA TextGrids")
    parser.add_argument("--diarize_dir", default="./data/processed", help="Folder with Speaker TextGrids")
    parser.add_argument("--output", default="./data/processed/yine_metrics.csv")
    args = parser.parse_args()
    
    process_corpus(args.txt_dir, args.wav_dir, args.mfa_dir, args.diarize_dir, args.output)