# Chatterbox-Streaming TTS
The orginal Chatterbox repository from Resemble AI can be found here: https://github.com/resemble-ai/chatterbox
The Chatterbox box streaming fork which provided much of the streaming framework can be found here: https://github.com/davidbrowne17/chatterbox-streaming 

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
To fine-tune Chatterbox all you need are some wav audio files with the speaker voice you want to train, just the raw wavs. Place them in the `audio_data` folder and run `lora.py`. You can configure the exact training params such as batch size, number of epochs and learning rate by modifying the values at the top of `lora.py`. You will need a CUDA gpu with at least 18gb of vram depending on your dataset size and training params. You can monitor the training metrics via the dynamic png created called `training_metrics.png`. This contains various graphs to help you track the training progress. Checkpoints and merged model will be saved to the a folder named `checkpoints_dir`. If you want to try a checkpoint you can use the `loadandmergecheckpoint.py` (make sure to set the same R and Alpha values as you used in the training). If you are using the TTS evaluation framework, once you have the path local path to your merged model assign it to `tuned_model_path` in `config.toml`.

## Streaming Parameter Tips
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

# Initialize and setup model
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
The original [chatterbox-streaming](https://github.com/davidbrowne17/chatterbox-streaming) only included a simple linear fade in to newly processed chunks. This implementation failed to properly elimiate the precense of "popping" noises or audio artifacts at chunk boundaries, and it made speech generated on small chunk sizes unusable. To attempt to remedy this I implemented equal part cross-fading between chunks. While this was a great improvment over the initial implemenation it failed to emliminate all audio artifacts at chunk boundaries. 

You can use either hann cross-fading or linear cross-fading, and you can modify which cross-faing type within the `_process_token_buffer()` method in the `tts.py` file. I found that using linear equal parts cross-fading with a fade_duration of 256 samples worked the best.

### Server/Client Proof of Concept
I added the `server.py` and `client.py` as server/client proof of concept, It makes use of multiprocessing to perform chunk generation and chunk processing in parrallel which, while doesn't directly reduce latency, reduces the overall total generation time which allows us incurr the increased overhead of using a smaller chunk size. To test the proof of concept, run `server.py` on one machine and wait for it to search for a connection. Once it's waiting on a connection, on another machine run the `client.py` file. The server side will generate and process chunks before sending them over a direct tcp connection in packages of 1024 bytes, and the client side will recieve the audio as raw bytes and pass it directly in a pyaudio stream for playback.
