# import torch
# import torchaudio
# print(torch.__version__)
# print(torchaudio.__version__)
# import numpy as np

# import IPython
# import matplotlib.pyplot as plt

# import torchaudio.functional as F

# # def process_audio_and_transcript(audio_file_path, transcript):
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

# SPEECH_FILE = ("abc.wav")
# waveform, _ = torchaudio.load(SPEECH_FILE)
# TRANSCRIPT = "the weather is nice today".split()

# bundle = torchaudio.pipelines.MMS_FA

# model = bundle.get_model(with_star=False).to(device)
# with torch.inference_mode():
#     emission, _ = model(waveform.to(device))

# def plot_emission(emission):
#     fig, ax = plt.subplots()
#     ax.imshow(emission.cpu().T)
#     ax.set_title("Frame-wise class probabilities")
#     ax.set_xlabel("Time")
#     ax.set_ylabel("Labels")
#     fig.tight_layout()

# #get tensors response

# plot_emission(emission[0])
# print(emission[0])

# LABELS = bundle.get_labels(star=None)
# DICTIONARY = bundle.get_dict(star=None)
# for k, v in DICTIONARY.items():
#     print(f"{k}: {v}")

# tokenized_transcript = [DICTIONARY[c] for word in TRANSCRIPT for c in word]

# for t in tokenized_transcript:
#     print(t, end=" ")
# print()

# def align(emission, tokens):
#     targets = torch.tensor([tokens], dtype=torch.int32, device=device)
#     alignments, scores = F.forced_align(emission, targets, blank=0)

#     alignments, scores = alignments[0], scores[0]  # remove batch dimension for simplicity
#     scores = scores.exp()  # convert back to probability
#     return alignments, scores

# aligned_tokens, alignment_scores = align(emission, tokenized_transcript)

# for i, (ali, score) in enumerate(zip(aligned_tokens, alignment_scores)):
#     print(f"{i:3d}:\t{ali:2d} [{LABELS[ali]}], {score:.2f}")

# token_spans = F.merge_tokens(aligned_tokens, alignment_scores)

# print("Token/tTime/tScore")
# for s in token_spans:
#     print(f"{LABELS[s.token]}\t[{s.start:3d}, {s.end:3d})\t{s.score:.2f}")

# def unflatten(list_, lengths):
#     assert len(list_) == sum(lengths)
#     i = 0
#     ret = []
#     for l in lengths:
#         ret.append(list_[i : i + l])
#         i += l
#     return ret


# word_spans = unflatten(token_spans, [len(word) for word in TRANSCRIPT])

# # Compute average score weighted by the span length
# def _score(spans):
#     return sum(s.score * len(s) for s in spans) / sum(len(s) for s in spans)


# def preview_word(waveform, spans, num_frames, transcript, sample_rate=bundle.sample_rate):
#     ratio = waveform.size(1) / num_frames
#     x0 = int(ratio * spans[0].start)
#     x1 = int(ratio * spans[-1].end)
#     print(f"{transcript} ({_score(spans):.2f}): {x0 / sample_rate:.3f} - {x1 / sample_rate:.3f} sec")
#     segment = waveform[:, x0:x1]
#     return IPython.display.Audio(segment.numpy(), rate=sample_rate)

# # Generate the audio for each segment
# print(TRANSCRIPT)
# IPython.display.Audio(SPEECH_FILE)

# def plot_alignments(waveform, token_spans, emission, transcript, sample_rate=bundle.sample_rate):
#     ratio = waveform.size(1) / emission.size(1) / sample_rate

#     fig, axes = plt.subplots(2, 1)
#     axes[0].imshow(emission[0].detach().cpu().T, aspect="auto")
#     axes[0].set_title("Emission")
#     axes[0].set_xticks([])

#     axes[1].specgram(waveform[0], Fs=sample_rate)
#     for t_spans, chars in zip(token_spans, transcript):
#         t0, t1 = t_spans[0].start + 0.1, t_spans[-1].end - 0.1
#         axes[0].axvspan(t0 - 0.5, t1 - 0.5, facecolor="None", hatch="/", edgecolor="white")
#         axes[1].axvspan(ratio * t0, ratio * t1, facecolor="None", hatch="/", edgecolor="white")
#         axes[1].annotate(f"{_score(t_spans):.2f}", (ratio * t0, sample_rate * 0.51), annotation_clip=False)

#         for span, char in zip(t_spans, chars):
#             t0 = span.start * ratio
#             axes[1].annotate(char, (t0, sample_rate * 0.55), annotation_clip=False)

#     axes[1].set_xlabel("time [second]")
#     axes[1].set_xlim([0, None])
#     fig.tight_layout()
# plot_alignments(waveform, word_spans, emission, TRANSCRIPT)
# plt.show()

# # print(waveform, word_spans, emission, TRANSCRIPT)





#add preprocess of audio file

import torch
import torchaudio
import matplotlib.pyplot as plt
import torchaudio.functional as F
from torchaudio.pipelines import MMS_FA
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import IPython.display as ipd

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Function to preprocess audio
def preprocess_audio(waveform, sample_rate, target_sample_rate=16000):
    # Resample if necessary
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
        sample_rate = target_sample_rate

    # Normalize the waveform
    waveform = waveform / waveform.abs().max()
    
    return waveform, sample_rate

# Function to ensure minimum segment length
def ensure_min_length(waveform, sample_rate, min_duration=0.5):
    min_length = int(min_duration * sample_rate)
    if waveform.size(1) < min_length:
        padding = min_length - waveform.size(1)
        waveform = torch.nn.functional.pad(waveform, (0, padding), mode='constant', value=0)
    return waveform

# Load the audio file and transcript
SPEECH_FILE = "recorded_audio (11).wav"
waveform, sample_rate = torchaudio.load(SPEECH_FILE)

# Preprocess the entire audio file
waveform, sample_rate = preprocess_audio(waveform, sample_rate)

# Transcript
TRANSCRIPT = "the weather is nice today".split()

# Load the MMS model and dictionary
bundle = MMS_FA
mms_model = bundle.get_model(with_star=False).to(device)
DICTIONARY = bundle.get_dict(star=None)

# Get the emission probabilities from the MMS model
with torch.inference_mode():
    emission, _ = mms_model(waveform.to(device))

# Tokenize the transcript
LABELS = bundle.get_labels(star=None)
tokenized_transcript = [DICTIONARY[c] for word in TRANSCRIPT for c in word]

# Align function
def align(emission, tokens):
    targets = torch.tensor([tokens], dtype=torch.int32, device=device)
    alignments, scores = F.forced_align(emission, targets, blank=0)
    alignments, scores = alignments[0], scores[0]  # remove batch dimension for simplicity
    scores = scores.exp()  # convert back to probability
    return alignments, scores

aligned_tokens, alignment_scores = align(emission, tokenized_transcript)

# Merge tokens into spans
token_spans = F.merge_tokens(aligned_tokens, alignment_scores)

# Unflatten list function
def unflatten(list_, lengths):
    assert len(list_) == sum(lengths)
    i = 0
    ret = []
    for l in lengths:
        ret.append(list_[i : i + l])
        i += l
    return ret

# Get word spans
word_spans = unflatten(token_spans, [len(word) for word in TRANSCRIPT])

# Compute average score weighted by the span length
def _score(spans):
    return sum(s.score * len(s) for s in spans) / sum(len(s) for s in spans)

# Function to preview word
def preview_word(waveform, spans, num_frames, transcript, sample_rate=bundle.sample_rate):
    ratio = waveform.size(1) / num_frames
    x0 = int(ratio * spans[0].start)
    x1 = int(ratio * spans[-1].end)
    print(f"{transcript} ({_score(spans):.2f}): {x0 / sample_rate:.3f} - {x1 / sample_rate:.3f} sec")
    segment = waveform[:, x0:x1]
    return segment

# Load the Wav2Vec2 processor and model for phoneme extraction
ctc_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
ctc_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft").to(device)

# Function to get phonemes
def get_phonemes(segment, processor, model, sample_rate):
    try:
        # Ensure segment is in the correct format
        if segment.dim() == 1:
            segment = segment.unsqueeze(0)
        if segment.dim() == 2 and segment.size(0) > 1:
            segment = segment.mean(dim=0, keepdim=True)
        
        # Ensure the segment has a minimum length
        segment = ensure_min_length(segment, sample_rate)

        # The model expects a tensor of shape (batch_size, sequence_length)
        input_values = processor(segment.squeeze(), return_tensors="pt", sampling_rate=sample_rate).input_values.to(device)
        
        # Forward pass
        with torch.no_grad():
            logits = model(input_values).logits
        
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)
        return transcription[0]
    except Exception as e:
        print(f"Error processing segment: {e}")
        return ""

# Generate word audio segments and get phonemes
word_audio_segments = []
word_phonemes = []

for spans, word in zip(word_spans, TRANSCRIPT):
    segment = preview_word(waveform, spans, emission.size(1), word)
    word_audio_segments.append(segment)
    phonemes = get_phonemes(segment, ctc_processor, ctc_model, sample_rate)
    word_phonemes.append(phonemes)

# Print phonemes for each word
for word, phonemes in zip(TRANSCRIPT, word_phonemes):
    print(f"{word}: {phonemes}")

# Function to listen to word segments
def listen_to_word_segments(word_audio_segments, word_phonemes, sample_rate):
    for i, (segment, phonemes) in enumerate(zip(word_audio_segments, word_phonemes)):
        print(f"Word {i}: {TRANSCRIPT[i]} - Phonemes: {phonemes}")
        ipd.display(ipd.Audio(segment.cpu().numpy(), rate=sample_rate))
        print()

# Listen to each word segment with phonemes
listen_to_word_segments(word_audio_segments, word_phonemes, sample_rate)
print(TRANSCRIPT)
print(word_phonemes)

# def plot_alignments(waveform, token_spans, emission, transcript, sample_rate=bundle.sample_rate):
#     ratio = waveform.size(1) / emission.size(1) / sample_rate

#     fig, axes = plt.subplots(2, 1)
#     axes[0].imshow(emission[0].detach().cpu().T, aspect="auto")
#     axes[0].set_title("Emission")
#     axes[0].set_xticks([])

#     axes[1].specgram(waveform[0], Fs=sample_rate)
#     for t_spans, chars in zip(token_spans, transcript):
#         t0, t1 = t_spans[0].start + 0.1, t_spans[-1].end - 0.1
#         axes[0].axvspan(t0 - 0.5, t1 - 0.5, facecolor="None", hatch="/", edgecolor="white")
#         axes[1].axvspan(ratio * t0, ratio * t1, facecolor="None", hatch="/", edgecolor="white")
#         axes[1].annotate(f"{_score(t_spans):.2f}", (ratio * t0, sample_rate * 0.51), annotation_clip=False)

#         for span, char in zip(t_spans, chars):
#             t0 = span.start * ratio
#             axes[1].annotate(char, (t0, sample_rate * 0.55), annotation_clip=False)

#     axes[1].set_xlabel("time [second]")
#     axes[1].set_xlim([0, None])
#     fig.tight_layout()
# plot_alignments(waveform, word_spans, emission, TRANSCRIPT)
# plt.show()
