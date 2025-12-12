# Chatterbox-Streaming TTS
The original Chatterbox repository from Resemble AI can be found here: https://github.com/resemble-ai/chatterbox

The Chatterbox streaming fork, which provided much of the streaming framework, can be found here: https://github.com/davidbrowne17/chatterbox-streaming 

## Key Details
- Multilingual, zero-shot TTS supporting 23 languages
- SoTA zeroshot English TTS
- 0.5B Llama backbone
- Unique exaggeration/intensity control
- Ultra-stable with alignment-informed inference
- Trained on 0.5M hours of cleaned data
- Watermarked outputs
- Easy voice conversion script

## LoRA Fine-Tuning
To fine-tune Chatterbox, all you need are some WAV audio files with the speaker voice you want to train, just the raw WAVs. Place them in the `audio_data` folder and run `lora.py`. You can configure the exact training parameters, such as batch size, number of epochs, and learning rate, by modifying the values at the top of `lora.py`. You will need a CUDA GPU with at least 18 GB of VRAM, depending on your dataset size and training parameters. You can monitor the training metrics via the dynamic PNG called `training_metrics.png`. This contains various graphs to help you track the training progress. Checkpoints and the merged model will be saved to a folder named `checkpoints_dir`. If you want to try a checkpoint, you can use the `loadandmergecheckpoint.py` (make sure to set the same R and Alpha values as you used in the training). If you are using the TTS evaluation framework, once you have the local path to your merged model, assign it to `tuned_model_path` in `config.toml`.

## Streaming Parameters
The Chatterbox model provides zero-shot voice cloning, allowing you to mimic any voice using a short audio sample. To provide an audio sample for Chatterbox to use, place your file in the `inputs/audio_pompts` directory and set Chatterbox's `audio_prompt` parameter to the file name. You can also control various aspects of the generated speech through the following parameters: 
- `exaggeration`: Emotion intensity control
  - Range: 0.0-1.0
  - Suggested Value: 0.5
- `cfg_weight`: Classifier-free guidance weight
  - Range: 0.0-1.0
  - Suggested Value: 0.3
- `chunk_size`: Number of speech tokens per chunk. Increased chunk_size will greatly improve quality, but leads to a significant increase in latency.
  - Suggested Value: 25
- `fade_duration`: Number of samples over which to apply equal part cross-fading
  - Suggested Value: 256
- `temperature`: Sampling randomness
  - Range: 0.1-0.9
  - Suggested Value: 0.1
### Streaming Parameter Tips
- **General Use (TTS and Voice Agents):**
  - Ensure that the reference clip matches the specified language tag. Otherwise, language transfer outputs may inherit the accent of the reference clip’s language. To mitigate this, set `cfg_weight` to `0`.
  - The default settings (`exaggeration=0.5`, `cfg_weight=0.5`) work well for most prompts across all languages.
  - If the reference speaker has a fast speaking style, lowering `cfg_weight` to around `0.3` can improve pacing.

- **Expressive or Dramatic Speech:**
  - Try lower `cfg_weight` values (e.g. `~0.3`) and increase `exaggeration` to around `0.7` or higher.
  - Higher `exaggeration` tends to speed up speech; reducing `cfg_weight` helps compensate with slower, more deliberate pacing.

## Modifications and Optimizations
### Streaming Setup
In reviewing the framework, there were many processes that could be brought outside the inference request method and precomputed. Now before calling `generate_stream()`, you must call the `setup_model()` method. Here is an example implementation:
```python
import torch
from chatterbox import ChatterboxTTS
import soundfile as sf
import numpy as np

# Initialize and set up the the  model
model = ChatterboxTTS.from_pretrained(device="cuda")
model.setup_model(
    audio_prompt_path="/audio/prompt/path",
    exaggeration=0.5,
    fade_duration=1024,
)

audio = []

# Perform Inference
for audio_chunk in self.model.generate_stream(
    text="Hi, I am an AI assistant! How can I help you today?",
    cfg_weight=0.5,
    chunk_size=25,
    temperature=0.8,
    context_window=25,
):
    audio.append(audio_chunk)

    # Reconstruct Audio
    audio = np.concatenate(audio, axis=0)
    sf.write("/output/audio/dir", audio, samplerate=24000, subtype="FLOAT")
```

### Equal Part Cross-Fading
The original [chatterbox-streaming](https://github.com/davidbrowne17/chatterbox-streaming) only included a simple linear fade in to newly processed chunks. This implementation failed to properly eliminate the presence of "popping" noises or audio artifacts at chunk boundaries, and it made speech generated on small chunk sizes unusable. To attempt to remedy this, I implemented equal parts cross-fading between chunks. While this was a great improvement over the initial implementation, it failed to eliminate all audio artifacts at chunk boundaries. 

You can use either Hann cross-fading or linear cross-fading, and you can modify which cross-fading type within the `_process_token_buffer()` method in the `tts.py` file. I found that using linear equal parts cross-fading with a fade_duration of 256 samples worked the best.

### Server/Client Proof of Concept
I added the `server.py` and `client.py` as a server/client proof of concept. It makes use of multiprocessing to perform chunk generation and chunk processing in parallel, which, while it  doesn't directly reduce latency, reduces the overall total generation time, which allows us to incur the increased overhead of using a smaller chunk size. To test the proof of concept, run `server.py` on one machine and wait for it to search for a connection. Once it's waiting on a connection, on another machine, run the `client.py` file. The server side will generate and process chunks before sending them over a direct TCP connection in packages of 1024 bytes, and the client side will receive the audio as raw bytes and pass it directly in a PyAudio stream for playback.
