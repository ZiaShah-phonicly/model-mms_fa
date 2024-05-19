# import soundfile as sf
# import numpy as np
# from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
# import torch

# # Load audio file
# ds = "the weather is nice today".split()
# audio_file = "abc.wav"
# audio, sample_rate = sf.read(audio_file)

# # Preprocess audio
# # Check if the audio is stereo (has two channels) and convert to mono if needed
# if audio.ndim > 1:
#     audio = np.mean(audio, axis=1)  # Convert stereo to mono by averaging channels

# processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
# input_values = processor(ds, audio, return_tensors="pt").input_values

# # Load model
# model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# # Inference
# with torch.no_grad():
#     logits = model(input_values).logits

# # Decode
# predicted_ids = torch.argmax(logits, dim=-1)
# transcription = processor.batch_decode(predicted_ids)
# print("Transcription:", transcription)


# # from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
# # from datasets import load_dataset
# # import torch

# # # load model and processor
# # processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
# # model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
    
# # # load dummy dataset and read soundfiles
# # ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")

# # # # tokenize
# # # input_values = processor(ds[0]["audio"]["array"], return_tensors="pt").input_values

# # # # retrieve logits
# # # with torch.no_grad():
# # #     logits = model(input_values).logits

# # # # take argmax and decode
# # # predicted_ids = torch.argmax(logits, dim=-1)
# # # transcription = processor.batch_decode(predicted_ids)
# # #  # => should give ['m ɪ s t ɚ k w ɪ l t ɚ ɪ z ð ɪ ɐ p ɑː s əl l ʌ v ð ə m ɪ d əl k l æ s ɪ z æ n d w iː aʊ ɡ l æ d t ə w ɛ l k ə m h ɪ z ɡ ɑː s p ə']






# from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
# import torch
# import soundfile as sf
# import os

# # Function to process and transcribe audio
# def transcribe_audio(model, processor, audio_file_path, expected_text):
#     # Load the audio file
#     audio_input, sample_rate = sf.read(audio_file_path)
    
#     # Tokenize audio input
#     input_values = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values
    
#     # Retrieve logits
#     with torch.no_grad():
#         logits = model(input_values).logits
    
#     # Take argmax and decode
#     predicted_ids = torch.argmax(logits, dim=-1)
#     transcription = processor.batch_decode(predicted_ids)
    
#     return transcription, expected_text

# # Load model and processor
# processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
# model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")

# # Directory containing your audio files
# audio_files = "abc.wav"
# output_directory = "Audio sample"

# # Define the words for each audio sample
# audio_to_words = {
#     "audio1.wav": "the weather is nice today",
#     # "audio2.wav": "another_word1 another_word2",
#     # Add more audio files and their corresponding words here
# }

# import os

# # Directory containing your audio files
# audio_files = "abc.wav"
# # Update this with the correct directory path

# # Check if the directory exists
# if os.path.exists(audio_files) and os.path.isdir(audio_files):
#     # Process each audio file in the directory
#     for audio_file_name in os.listdir(audio_files):
#         if audio_file_name.endswith(".wav"):  # Ensure we're only processing .wav files
#             audio_file_path = os.path.join(audio_files, audio_file_name)
            
#             # Get expected text for current audio file
#             expected_text = audio_to_words.get(audio_file_name, "No words provided for this sample.")
            
#             # Transcribe audio
#             transcription, expected_text = transcribe_audio(model, processor, audio_file_path, expected_text)
            
#             # Save transcription and expected text to a file
#             transcription_file_path = os.path.join(output_directory, f"{os.path.splitext(audio_file_name)[0]}_transcription.txt")
#             with open(transcription_file_path, "w") as f:
#                 f.write(f"Transcription:\n{transcription[0]}\n\nExpected Text:\n{expected_text}")
            
#             print(f"Processed {audio_file_name}. Transcription and expected text saved to {transcription_file_path}")
# else:
#     print(f"The directory '{audio_files}' does not exist or is not a valid directory.")





# from pydub import AudioSegment
# from pydub.silence import split_on_silence
# import os

# # Path to your input audio file
# input_audio_path = "abc.wav"

# # Directory to save the sample audio files
# output_directory = "Audio sample"
# os.makedirs(output_directory, exist_ok=True)

# # Load the audio file
# audio = AudioSegment.from_wav(input_audio_path)

# # Split the audio where silence is 500ms or more and get chunks
# chunks = split_on_silence(audio, min_silence_len=500, silence_thresh=audio.dBFS-14, keep_silence=150)

# # Save each chunk as a separate file
# for i, chunk in enumerate(chunks):
#     chunk.export(os.path.join(output_directory, f"word_{i}.wav"), format="wav")
#     print(f"Generated sample word_{i}.wav")


from pydub import AudioSegment
from pydub.silence import split_on_silence
import os

# Path to your input audio file
input_audio = "Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"
# Directory to save the sample audio files
output_directory = "Audio sample"
os.makedirs(output_directory, exist_ok=True)

# Load the audio file with error handling
try:
    audio = AudioSegment.from_wav(input_audio)
except FileNotFoundError:
    print(f"Error: The file {input_audio} was not found.")
    exit(1)
except Exception as e:
    print(f"An error occurred while loading the audio file: {e}")
    exit(1)

# Split the audio where silence is 500ms or more and get chunks
chunks = split_on_silence(audio, min_silence_len=500, silence_thresh=audio.dBFS-14, keep_silence=150)

# Save each chunk as a separate file
for i, chunk in enumerate(chunks):
    chunk.export(os.path.join(output_directory, f"word_{i}.wav"), format="wav")
    print(f"Generated sample word_{i}.wav")
