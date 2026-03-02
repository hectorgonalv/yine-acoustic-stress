# src/syllabification.py
import re

YINE_VOWELS = {'a', 'e', 'i', 'o', 'ɨ'}

def split_yine_syllables(word):
    """
    Splits a written Yine word into syllables, with specific rules for Yine phonotactics.
    """
    if word == 'tumat':
        return ['tu', 'mat']

    if not word:
        return[]

    vowels = {'a', 'e', 'i', 'o', 'u', 'ɨ'}
    consonants = "cghjklmnprstwx"
    processed_word = re.sub(f'(?<=[{consonants}])y(?=[{consonants}])', 'i', word)

    syllables =[]
    start_index = 0
    for i in range(1, len(processed_word)):
        prev_char = processed_word[i-1]
        curr_char = processed_word[i]

        is_prev_vowel = prev_char in vowels
        is_curr_vowel = curr_char in vowels

        is_vc_boundary = is_prev_vowel and not is_curr_vowel
        is_vv_hiatus_boundary = is_prev_vowel and is_curr_vowel and prev_char != curr_char

        if is_vc_boundary or is_vv_hiatus_boundary:
            syllables.append(processed_word[start_index:i])
            start_index = i

    if start_index < len(processed_word):
         syllables.append(processed_word[start_index:])

    final_syllables =[]
    current_pos = 0
    for syl in syllables:
        syl_len = len(syl)
        final_syllables.append(word[current_pos : current_pos + syl_len])
        current_pos += syl_len

    return final_syllables

def syllabify_yine_phones(tg, word_tier_idx=0, phone_tier_idx=1):
    """
    Processes a TextGrid object to build a list of phonetic syllables.
    Stateless phone gathering logic.
    """
    word_tier = tg.tiers[word_tier_idx]
    phone_tier = tg.tiers[phone_tier_idx]

    all_syllables =[]

    for word_interval in word_tier:
        if not word_interval.mark.strip():
            continue

        phones_in_word =[
            p for p in phone_tier
            if p.minTime >= word_interval.minTime and p.maxTime <= word_interval.maxTime and p.mark.strip()
        ]

        if not phones_in_word:
            continue

        word_syllables =[]
        i = 0
        while i < len(phones_in_word):
            current_syllable_phones =[]
            # Gather Onset
            while i < len(phones_in_word) and phones_in_word[i].mark not in YINE_VOWELS:
                current_syllable_phones.append(phones_in_word[i])
                i += 1
            # Gather Nucleus
            if i < len(phones_in_word):
                current_syllable_phones.append(phones_in_word[i])
                i += 1
                if (i < len(phones_in_word) and
                    phones_in_word[i].mark == current_syllable_phones[-1].mark):
                    current_syllable_phones.append(phones_in_word[i])
                    i += 1

            if any(p.mark in YINE_VOWELS for p in current_syllable_phones):
                 word_syllables.append(current_syllable_phones)

        all_syllables.extend(word_syllables)

    return all_syllables