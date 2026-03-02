# scripts/05_prep_statistics.py
import argparse
import os
import numpy as np
import pandas as pd

# --- Stress Level Constants ---
PRIMARY_STRESS = "Primary_stress"
SECONDARY_STRESS = "Secondary_stress"
UNSTRESSED = "Unstressed"

def clean_speaker_labels(df):
    print("--- 1. Cleaning Speaker Labels ---")
    word_instance_cols =['File_Name', 'Utterance_id', 'Word_Index_in_Utt']

    def get_valid_speaker(speaker_series):
        valid_speakers = speaker_series[speaker_series != 'silence'].unique()
        return valid_speakers[0] if len(valid_speakers) > 0 else 'silence'

    corrected_speakers = df.groupby(word_instance_cols, dropna=False)['Speaker'].transform(get_valid_speaker)
    num_corrected = (df['Speaker'] != corrected_speakers).sum()
    print(f"  -> Fixed {num_corrected} incorrect 'silence' labels.")
    
    df['Speaker'] = corrected_speakers
    return df

def identify_problematic_data(df):
    word_instance_cols =['File_Name', 'Utterance_id', 'Word_Index_in_Utt']

    # 1. Multi-speaker utterances
    unique_speakers_per_utt = df.groupby('Utterance_id')['Speaker'].unique()
    problematic_utts = unique_speakers_per_utt[
        unique_speakers_per_utt.apply(lambda s: len([sp for sp in s if sp != 'silence']) > 1)
    ]
    utterance_ids_to_remove = problematic_utts.index.tolist()

    # 2. Multi-speaker words
    unique_speakers_per_word = df.groupby(word_instance_cols, dropna=False)['Speaker'].unique()
    multi_speaker_words = unique_speakers_per_word[
        unique_speakers_per_word.apply(lambda s: len([sp for sp in s if sp != 'silence']) > 1)
    ]
    
    # 3. Unfixable silence words
    unfixable_silence_words = unique_speakers_per_word[
        unique_speakers_per_word.apply(lambda s: all(sp == 'silence' for sp in s))
    ]
    
    word_indices_to_remove = list(set(multi_speaker_words.index.tolist() + unfixable_silence_words.index.tolist()))
    return utterance_ids_to_remove, word_indices_to_remove

def filter_problematic_data(df, utterance_ids_to_remove, word_indices_to_remove):
    print("--- 2. Filtering Problematic Data ---")
    original_rows = len(df)

    if utterance_ids_to_remove:
        df = df[~df['Utterance_id'].isin(utterance_ids_to_remove)].copy()
    
    if word_indices_to_remove:
        # Convert to MultiIndex for efficient filtering
        word_multi_index = pd.MultiIndex.from_tuples(word_indices_to_remove, names=['File_Name', 'Utterance_id', 'Word_Index_in_Utt'])
        df = df[~df.set_index(['File_Name', 'Utterance_id', 'Word_Index_in_Utt']).index.isin(word_multi_index)]

    print(f"  -> Removed {original_rows - len(df)} problematic rows (drift/silence).")
    return df.reset_index(drop=True)

def filter_monosyllables(df):
    print("--- 3. Filtering Monosyllabic Words ---")
    original_rows = len(df)
    df['Word_Syllable_Count'] = pd.to_numeric(df['Word_Syllable_Count'], errors='coerce')
    df_filtered = df[df['Word_Syllable_Count'] > 1].copy()
    print(f"  -> Removed {original_rows - len(df_filtered)} monosyllabic rows.")
    return df_filtered.reset_index(drop=True)

def filter_zero_acoustic_values(df):
    print("--- 4. Filtering Invalid Acoustics ---")
    original_rows = len(df)
    acoustic_cols_to_check =['Pitch_Q2', 'F1_mid', 'F2_mid']
    
    # Ensure numeric types
    for col in acoustic_cols_to_check:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    mask = (df[acoustic_cols_to_check] == 0).any(axis=1)
    df_filtered = df[~mask].copy()
    print(f"  -> Removed {original_rows - len(df_filtered)} rows with '0' in Pitch/Formants.")
    return df_filtered.reset_index(drop=True)

def annotate_matteson_stress(df):
    print("--- 5. Annotating Matteson's Stress Rules ---")
    df['Matteson_Stress'] = UNSTRESSED
    word_groups = df.groupby(['Utterance_id', 'Words'], sort=False)
    stress_annotations =[]

    for _, word_df in word_groups:
        num_syllables = len(word_df)
        syllable_stresses = [UNSTRESSED] * num_syllables

        # Rule 1: Primary Stress (Penultimate)
        if num_syllables >= 2:
            syllable_stresses[-2] = PRIMARY_STRESS

        # Rule 2: Secondary Stress (Initial for 4+)
        if num_syllables >= 4:
            if syllable_stresses[0] != PRIMARY_STRESS:
                 syllable_stresses[0] = SECONDARY_STRESS

        # Rule 3: Secondary Stress (Odd-numbered for 6+)
        if num_syllables >= 6:
            for i in range(0, num_syllables - 2, 2):
                is_odd_length_word = num_syllables % 2 != 0
                is_penultimate_of_secondary = (i == num_syllables - 3)

                if is_odd_length_word and is_penultimate_of_secondary:
                    continue 

                if syllable_stresses[i] == UNSTRESSED:
                    syllable_stresses[i] = SECONDARY_STRESS

        stress_annotations.extend(syllable_stresses)

    df['Matteson_Stress'] = stress_annotations
    return df

def preprocess_predictors_for_modeling(df):
    print("--- 6. Preprocessing Predictors (Log & Z-Score) ---")
    
    cols_to_clean =['Pitch_Q2', 'F1_mid', 'F2_mid']
    for col in cols_to_clean:
        df[col] = df[col].replace(0, np.nan)

    cols_to_log =['Duration', 'Pitch_Q2', 'F1_mid', 'F2_mid']
    for col in cols_to_log:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        new_col_name = f"Log_{col.split('_')[0]}"
        df[new_col_name] = np.log(df[col])

    cols_to_standardize =[
        'Log_Duration', 'Log_Pitch', 'Intensity_Q2', 'Spectral_Tilt_DB',
        'Log_F1', 'Log_F2', 'Pace_SPS', 'Syllable_position', 'Syllable_Index',
        'Word_Index_in_Utt', 'Word_Index_Reversed'
    ]

    for col in cols_to_standardize:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        mean = df[col].mean()
        std = df[col].std()
        base_name = col.replace('Log_', '')
        new_col_name = f"{base_name}_z"
        df[new_col_name] = (df[col] - mean) / std

    return df

def finalize_dataframe(df):
    print("--- 7. Finalizing DataFrame ---")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    df[numeric_cols] = df[numeric_cols].round(4)
    return df

def process_pipeline(input_csv, output_csv):
    if not os.path.exists(input_csv):
        print(f"Error: Input file {input_csv} not found.")
        return

    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)
    print(f"Started with {len(df)} rows.")

    # Execute Pipeline
    df = clean_speaker_labels(df)
    utts_to_remove, words_to_remove = identify_problematic_data(df)
    df = filter_problematic_data(df, utts_to_remove, words_to_remove)
    df = filter_monosyllables(df)
    
    # Annotate structure BEFORE dropping individual syllables ---
    df = annotate_matteson_stress(df)
    
    # Drop rows with bad acoustics ---
    df = filter_zero_acoustic_values(df)
    
    df = preprocess_predictors_for_modeling(df)
    df = finalize_dataframe(df)

    # Save Output
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    
    print("\n--- Summary ---")
    print(f"Successfully processed and saved {len(df)} modeling-ready rows to {output_csv}")
    print("\nSample of final data (Stress vs Z-scored Pitch):")
    print(df[['Words', 'Matteson_Stress', 'Duration_z', 'Pitch_z']].head(10).to_string())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare acoustic metrics for statistical modeling")
    parser.add_argument("--input_csv", default="./data/processed/yine_metrics.csv")
    parser.add_argument("--output_csv", default="./data/processed/yine_stats_metrics.csv")
    
    args = parser.parse_args()
    process_pipeline(args.input_csv, args.output_csv)