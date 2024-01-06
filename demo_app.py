import gradio as gr
import librosa
import numpy as np
import torch

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan


checkpoint = "microsoft/speecht5_tts"
processor = SpeechT5Processor.from_pretrained(checkpoint)
model = SpeechT5ForTextToSpeech.from_pretrained(checkpoint)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")


speaker_embeddings = {
    "BDL": "Speakers/cmu_us_bdl_arctic-wav-arctic_a0009.npy",
    "CLB": "Speakers/cmu_us_clb_arctic-wav-arctic_a0144.npy",
    "KSP": "Speakers/cmu_us_ksp_arctic-wav-arctic_b0087.npy",
    "RMS": "Speakers/cmu_us_rms_arctic-wav-arctic_b0353.npy",
    "SLT": "Speakers/cmu_us_slt_arctic-wav-arctic_a0508.npy",
}


def predict(text, speaker):
    if len(text.strip()) == 0:
        return (16000, np.zeros(0).astype(np.int16))

    inputs = processor(text=text, return_tensors="pt")

    # limit input length
    input_ids = inputs["input_ids"]
    input_ids = input_ids[..., :model.config.max_text_positions]

    if speaker == "Surprise Me!":
        # load one of the provided speaker embeddings at random
        idx = np.random.randint(len(speaker_embeddings))
        key = list(speaker_embeddings.keys())[idx]
        speaker_embedding = np.load(speaker_embeddings[key])

        # randomly shuffle the elements
        np.random.shuffle(speaker_embedding)

        # randomly flip half the values
        x = (np.random.rand(512) >= 0.5) * 1.0
        x[x == 0] = -1.0
        speaker_embedding *= x

        #speaker_embedding = np.random.rand(512).astype(np.float32) * 0.3 - 0.15
    else:
        speaker_embedding = np.load(speaker_embeddings[speaker[:3]])

    speaker_embedding = torch.tensor(speaker_embedding).unsqueeze(0)

    speech = model.generate_speech(input_ids, speaker_embedding, vocoder=vocoder)

    speech = (speech.numpy() * 32767).astype(np.int16)
    return (16000, speech)

title = "LoreWeaver: A Novel Generation Multimodal LLM"

gr.Interface(
    fn=predict,
    inputs=[
        gr.Text(label="Input Text"),
        gr.Radio(label="Speaker", choices=[
            "BDL (male)",
            "CLB (female)",
            "KSP (male)",
            "RMS (male)",
            "SLT (female)",
            "Surprise Me!"
        ],
        value="BDL (male)"),
    ],
    outputs=[
        gr.Audio(label="Generated Speech", type="numpy"),
    ],
    title=title,
).launch()

