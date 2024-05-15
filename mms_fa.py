# # # import torch  # Add this line to import the torch module
# # # import torchaudio
# # # import matplotlib.pyplot as plt

# # # # Load the pre-trained MMS_FA model
# # # path_to_mms_fa_model = 'C:/abcde/project_name/model.pt'
# # # mms_fa_model = torch.load(path_to_mms_fa_model)


# # # audio_file_path = 'model mms_fa/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav'
# # # text = 'I am Zia Shah'
# # # def align_with_mms_fa_model(audio_file_path, text):
# # #     # Load the audio file
# # #     waveform, sample_rate = torchaudio.load(audio_file_path, normalize=True, format='wav')
# # #     # Preprocess the waveform if needed
# # #     # For example, resample, convert to spectrogram

# # #     # Convert transcript to tensor
# # #     text_tensor = torch.tensor(text)

# # #     # Generate emissions
# # #     emissions = generate_emissions(waveform)

# # #     # Calculate alignment scores
# # #     alignment_scores = calculate_alignment_scores(emissions, text_tensor)

# # #     # Decode alignment scores
# # #     alignment_result = decode_alignment(alignment_scores)

# # #     # Plot alignments
# # #     plot_alignments(waveform, alignment_result, emissions, text_tensor, sample_rate)

# # #     return alignment_result

# # # def generate_emissions(waveform):
# # #     # Preprocess the waveform if needed
# # #     # For example, resample, convert to spectrogram

# # #     # Pass the preprocessed waveform through the model
# # #     with torch.no_grad():
# # #         emissions = mms_fa_model(waveform)

# # #     return emissions

# # # def calculate_alignment_scores(emissions, transcript_tensor):
# # #     # Calculate CTC loss using the emissions and the transcript tensor
# # #     ctc_loss = torch.nn.CTCLoss(blank=0)  # Assuming blank token index is 0
# # #     alignment_scores = ctc_loss(emissions, transcript_tensor)

# # #     return alignment_scores

# # # def decode_alignment(alignment_scores):
# # #     # Perform decoding to obtain the alignment result
# # #     # For simplicity, let's use argmax decoding
# # #     alignment_indices = torch.argmax(alignment_scores, dim=-1)

# # #     # Convert indices to text
# # #     alignment_result = ''.join([str(idx.item()) for idx in alignment_indices])

# # #     return alignment_result

# # # def plot_alignments(waveform, alignment_result, emissions, transcript_tensor, sample_rate):
# # #     ratio = waveform.size(1) / emissions.size(1) / sample_rate

# # #     fig, axes = plt.subplots(2, 1, figsize=(10, 6))

# # #     # Plot emission
# # #     axes[0].imshow(emissions[0].detach().cpu().T, aspect="auto", origin="lower")
# # #     axes[0].set_title("Emission")
# # #     axes[0].set_ylabel("Frames")

# # #     # Plot waveform spectrogram
# # #     axes[1].specgram(waveform[0], Fs=sample_rate)
# # #     axes[1].set_title("Waveform Spectrogram")
# # #     axes[1].set_ylabel("Frequency [Hz]")
# # #     axes[1].set_xlabel("Time [s]")

# # #     # Plot alignment result
# # #     for char_index, char in enumerate(alignment_result):
# # #         t0 = char_index * ratio
# # #         axes[1].text(t0, sample_rate * 0.9, char, fontsize=10, ha='center', va='center', color='black')

# # #     fig.tight_layout()
# # #     plt.show()
# # # try:
# # #     # Call the align_with_mms_fa_model function
# # #     alignment_result = align_with_mms_fa_model(audio_file_path, text)
# # #     print("Alignment result:", alignment_result)
# # # except Exception as e:
# # #     print("An error occurred:", e)

# # # # # Example usage
# # # # audio_file_path = 'path_to_audio_file.wav'
# # # # text = 'sample text for alignment'
# # # # alignment_result = align_with_mms_fa_model(audio_file_path, text)



# # import torch  # Add this line to import the torch module
# # import torchaudio
# # import matplotlib.pyplot as plt

# # # Load the pre-trained MMS_FA model
# # path_to_mms_fa_model = 'C:/abcde/project_name/model.pt'
# # mms_fa_model = torch.load(path_to_mms_fa_model)

# # audio_file_path = 'model mms_fa/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav'
# # text = "i had that curiosity beside me at this moment".split()

# # def align_with_mms_fa_model(audio_file_path, text):
# #     try:
# #         # Load the audio file
# #         waveform, sample_rate = torchaudio.load(audio_file_path, normalize=True, format='wav')
        
# #         # Preprocess the waveform if needed
# #         # For example, resample, convert to spectrogram

# #         # Convert transcript to tensor
# #         text_tensor = torch.tensor(text)

# #         # Generate emissions
# #         emissions = generate_emissions(waveform)

# #         # Calculate alignment scores
# #         alignment_scores = calculate_alignment_scores(emissions, text_tensor)

# #         # Decode alignment scores
# #         alignment_result = decode_alignment(alignment_scores)

# #         # Plot alignments
# #         plot_alignments(waveform, alignment_result, emissions, text_tensor, sample_rate)

# #         return alignment_result

# #     except Exception as e:
# #         print("An error occurred:", e)
# #         return None

# # def generate_emissions(waveform):
# #     # Preprocess the waveform if needed
# #     # For example, resample, convert to spectrogram

# #     # Pass the preprocessed waveform through the model
# #     with torch.no_grad():
# #         emissions = mms_fa_model(waveform)

# #     return emissions

# # def calculate_alignment_scores(emissions, transcript_tensor):
# #     # Calculate CTC loss using the emissions and the transcript tensor
# #     ctc_loss = torch.nn.CTCLoss(blank=0)  # Assuming blank token index is 0
# #     alignment_scores = ctc_loss(emissions, transcript_tensor)

# #     return alignment_scores

# # def decode_alignment(alignment_scores):
# #     try:
# #         # Perform decoding to obtain the alignment result
# #         # For simplicity, let's use argmax decoding
# #         alignment_indices = torch.argmax(alignment_scores, dim=-1)

# #         # Convert indices to text
# #         alignment_result = ''.join([str(idx.item()) for idx in alignment_indices])

# #         return alignment_result
# #     except Exception as e:
# #         print("An error occurred during decoding alignment scores:", e)
# #         return None


# # def plot_alignments(waveform, alignment_result, emissions, transcript_tensor, sample_rate):
# #     ratio = waveform.size(1) / emissions.size(1) / sample_rate

# #     fig, axes = plt.subplots(2, 1, figsize=(10, 6))

# #     # Plot emission
# #     axes[0].imshow(emissions[0].detach().cpu().T, aspect="auto", origin="lower")
# #     axes[0].set_title("Emission")
# #     axes[0].set_ylabel("Frames")

# #     # Plot waveform spectrogram
# #     axes[1].specgram(waveform[0], Fs=sample_rate)
# #     axes[1].set_title("Waveform Spectrogram")
# #     axes[1].set_ylabel("Frequency [Hz]")
# #     axes[1].set_xlabel("Time [s]")

# #     # Plot alignment result
# #     for char_index, char in enumerate(alignment_result):
# #         t0 = char_index * ratio
# #         axes[1].text(t0, sample_rate * 0.9, char, fontsize=10, ha='center', va='center', color='black')

# #     fig.tight_layout()
# #     plt.show()

# # # Call the align_with_mms_fa_model function
# # alignment_result = align_with_mms_fa_model(audio_file_path, text)
# # print("Alignment result:", alignment_result)


import torch
import torchaudio
print(torch.__version__)
print(torchaudio.__version__)

import IPython
import matplotlib.pyplot as plt

import torchaudio.functional as F

# def process_audio_and_transcript(audio_file_path, transcript):
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

SPEECH_FILE = ("abc.wav")
waveform, _ = torchaudio.load(SPEECH_FILE)
TRANSCRIPT = "the weather is nice today".split()

bundle = torchaudio.pipelines.MMS_FA

model = bundle.get_model(with_star=False).to(device)
with torch.inference_mode():
    emission, _ = model(waveform.to(device))

def plot_emission(emission):
    fig, ax = plt.subplots()
    ax.imshow(emission.cpu().T)
    ax.set_title("Frame-wise class probabilities")
    ax.set_xlabel("Time")
    ax.set_ylabel("Labels")
    fig.tight_layout()


plot_emission(emission[0])
print(emission[0])

LABELS = bundle.get_labels(star=None)
DICTIONARY = bundle.get_dict(star=None)
for k, v in DICTIONARY.items():
    print(f"{k}: {v}")

tokenized_transcript = [DICTIONARY[c] for word in TRANSCRIPT for c in word]

for t in tokenized_transcript:
    print(t, end=" ")
print()

def align(emission, tokens):
    targets = torch.tensor([tokens], dtype=torch.int32, device=device)
    alignments, scores = F.forced_align(emission, targets, blank=0)

    alignments, scores = alignments[0], scores[0]  # remove batch dimension for simplicity
    scores = scores.exp()  # convert back to probability
    return alignments, scores

aligned_tokens, alignment_scores = align(emission, tokenized_transcript)

for i, (ali, score) in enumerate(zip(aligned_tokens, alignment_scores)):
    print(f"{i:3d}:\t{ali:2d} [{LABELS[ali]}], {score:.2f}")

token_spans = F.merge_tokens(aligned_tokens, alignment_scores)

print("Token/tTime/tScore")
for s in token_spans:
    print(f"{LABELS[s.token]}\t[{s.start:3d}, {s.end:3d})\t{s.score:.2f}")

def unflatten(list_, lengths):
    assert len(list_) == sum(lengths)
    i = 0
    ret = []
    for l in lengths:
        ret.append(list_[i : i + l])
        i += l
    return ret


word_spans = unflatten(token_spans, [len(word) for word in TRANSCRIPT])

# Compute average score weighted by the span length
def _score(spans):
    return sum(s.score * len(s) for s in spans) / sum(len(s) for s in spans)


def preview_word(waveform, spans, num_frames, transcript, sample_rate=bundle.sample_rate):
    ratio = waveform.size(1) / num_frames
    x0 = int(ratio * spans[0].start)
    x1 = int(ratio * spans[-1].end)
    print(f"{transcript} ({_score(spans):.2f}): {x0 / sample_rate:.3f} - {x1 / sample_rate:.3f} sec")
    segment = waveform[:, x0:x1]
    return IPython.display.Audio(segment.numpy(), rate=sample_rate)

# Generate the audio for each segment
print(TRANSCRIPT)
IPython.display.Audio(SPEECH_FILE)

def plot_alignments(waveform, token_spans, emission, transcript, sample_rate=bundle.sample_rate):
    ratio = waveform.size(1) / emission.size(1) / sample_rate

    fig, axes = plt.subplots(2, 1)
    axes[0].imshow(emission[0].detach().cpu().T, aspect="auto")
    axes[0].set_title("Emission")
    axes[0].set_xticks([])

    axes[1].specgram(waveform[0], Fs=sample_rate)
    for t_spans, chars in zip(token_spans, transcript):
        t0, t1 = t_spans[0].start + 0.1, t_spans[-1].end - 0.1
        axes[0].axvspan(t0 - 0.5, t1 - 0.5, facecolor="None", hatch="/", edgecolor="white")
        axes[1].axvspan(ratio * t0, ratio * t1, facecolor="None", hatch="/", edgecolor="white")
        axes[1].annotate(f"{_score(t_spans):.2f}", (ratio * t0, sample_rate * 0.51), annotation_clip=False)

        for span, char in zip(t_spans, chars):
            t0 = span.start * ratio
            axes[1].annotate(char, (t0, sample_rate * 0.55), annotation_clip=False)

    axes[1].set_xlabel("time [second]")
    axes[1].set_xlim([0, None])
    fig.tight_layout()
plot_alignments(waveform, word_spans, emission, TRANSCRIPT)
plt.show()

# print(waveform, word_spans, emission, TRANSCRIPT)