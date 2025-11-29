from diffusers import AudioLDM2Pipeline
import torch
import scipy.io.wavfile as wavfile

def main():
    model_id = "cvssp/audioldm2"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the pipeline with half-precision (optional if on GPU)
    pipe = AudioLDM2Pipeline.from_pretrained(model_id, torch_dtype=torch.float16 if device=="cuda" else torch.float32)
    pipe = pipe.to(device)

    prompt = "High-quality sound of water being poured into a glass in a quiet room"
    audio_length = 5.0  # seconds
    steps = 200

    outputs = pipe(prompt, num_inference_steps=steps, audio_length_in_s=audio_length)
    audio = outputs.audios[0]

    wavfile.write("pouring_water.wav", rate=16000, data=audio)
    print("Saved to pouring_water.wav")

if __name__ == "__main__":
    main()
