from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import soundfile as sf

# Load model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")

# Dummy audio samples and corresponding text (replace with your actual data)
audio_samples = [
    {"path": "download (1).wav", "text": "curiosity"}
    # {"path": "path/to/sample2.wav", "text": "word2"},
    # {"path": "path/to/sample3.wav", "text": "word3"},
    # {"path": "path/to/sample4.wav", "text": "word4"},
    # {"path": "path/to/sample5.wav", "text": "word5"}
]

# Iterate over each sample
for sample in audio_samples:
    audio_path = sample["path"]
    expected_text = sample["text"]
    
    # Load the audio file
    audio_input, sample_rate = sf.read(audio_path)
    
    # Tokenize audio input
    input_values = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values
    
    # Retrieve logits
    with torch.no_grad():
        logits = model(input_values).logits
    
    # Take argmax and decode
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    
    # Print transcription and expected text
    print(f"Transcription for {audio_path}: {transcription[0]}")
    print(f"Expected Text: {expected_text}\n")
    print(transcription)
