# scripts/03_diarize_audio.py
import argparse
import os
import torch
import librosa
from pyannote.audio import Pipeline
from textgrid import TextGrid, IntervalTier

def create_speaker_textgrid(diarization, audio_duration, output_path, base_filename):
    """
    Converts pyannote diarization output to a TextGrid.
    Handles overlaps by flattening the timeline and combining labels.
    """
    tg = TextGrid(minTime=0, maxTime=audio_duration)
    speaker_tier = IntervalTier(name="speaker", minTime=0, maxTime=audio_duration)

    all_time_points = {0.0, audio_duration}
    for turn, _, _ in diarization.itertracks(yield_label=True):
        all_time_points.add(turn.start)
        all_time_points.add(turn.end)
    
    sorted_points = sorted(list(all_time_points))

    for i in range(len(sorted_points) - 1):
        start_time = sorted_points[i]
        end_time = sorted_points[i+1]
        midpoint = start_time + (end_time - start_time) / 2
        speakers_at_midpoint = diarization.get_labels(midpoint)

        if not speakers_at_midpoint:
            label = "silence"
        elif len(speakers_at_midpoint) == 1:
            label = f"{base_filename}_{speakers_at_midpoint[0]}"
        else:
            sorted_labels = sorted(speakers_at_midpoint)
            global_labels = [f"{base_filename}_{lbl}" for lbl in sorted_labels]
            label = "+".join(global_labels)

        speaker_tier.add(start_time, end_time, label)

    tg.append(speaker_tier)
    tg.write(output_path)

def process_corpus(input_dir, output_dir):
    """Main processing loop."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Initializing Pyannote Pipeline...")
    
    # PyTorch 2.6 defaults weights_only=True, which blocks pyannote models.
    # This temporarily forces it to False to allow the model to load.
    try:
        torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
    except AttributeError:
        pass

    _original_load = torch.load
    def _force_safe_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return _original_load(*args, **kwargs)

    torch.load = _force_safe_load

    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    except Exception as e:
        print("\n[!] Error loading Pyannote model.")
        print("    Please ensure you have accepted the user conditions on Hugging Face")
        print("    and are correctly logged in via 'hf auth login'.")
        print(f"    Details: {e}")
        return
    finally:
        # Restore the original torch.load immediately after loading the pipeline
        torch.load = _original_load

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    pipeline.to(device)

    audio_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
    print(f"Found {len(audio_files)} audio files.")

    for filename in audio_files:
        audio_path = os.path.join(input_dir, filename)
        base_name = os.path.splitext(filename)[0]
        output_tg_path = os.path.join(output_dir, f"{base_name}_speaker.TextGrid")

        print(f"Processing: {filename}...")
        
        try:
            y, sr = librosa.load(audio_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            diarization = pipeline(audio_path)
            create_speaker_textgrid(diarization, duration, output_tg_path, base_name)
            print(f"  [V] Saved TextGrid.")
        except Exception as e:
            print(f"  [X] Failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Speaker Diarization on audio corpus")
    parser.add_argument("--input_dir", required=True, help="Folder containing .wav files")
    parser.add_argument("--output_dir", required=True, help="Folder to save .TextGrid files")
    
    args = parser.parse_args()
    
    process_corpus(args.input_dir, args.output_dir)