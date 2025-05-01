import argparse
import logging
import os
import re
import gc

import faster_whisper
import torch
import torchaudio

from ctc_forced_aligner import (
    generate_emissions,
    get_alignments,
    get_spans,
    load_alignment_model,
    postprocess_results,
    preprocess_text,
)
from deepmultilingualpunctuation import PunctuationModel
from nemo.collections.asr.models.msdd_models import NeuralDiarizer

from df import enhance, init_df
from df.io import load_audio, save_audio

from helpers import (
    cleanup,
    create_config,
    create_config_custom,
    load_config,
    find_numeral_symbol_tokens,
    get_realigned_ws_mapping_with_punctuation,
    get_sentences_speaker_mapping,
    get_speaker_aware_transcript,
    get_words_speaker_mapping,
    langs_to_iso,
    process_language_arg,
    punct_model_langs,
    whisper_langs,
    write_srt,
)

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)

mtypes = {"cpu": "int8", "cuda": "float16"}

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "-a", "--audio", help="name of the target audio file", required=True
)
parser.add_argument(
    "--no-stem",
    action="store_false",
    dest="stemming",
    default=True,
    help="Disables source separation."
         "This helps with long files that don't contain a lot of music.",
)

parser.add_argument(
    "--suppress_numerals",
    action="store_true",
    dest="suppress_numerals",
    default=False,
    help="Suppresses Numerical Digits."
         "This helps the diarization accuracy but converts all digits into written text.",
)

parser.add_argument(
    "--whisper-model",
    dest="model_name",
    default="medium.en",
    help="name of the Whisper model to use",
)

parser.add_argument(
    "--batch-size",
    type=int,
    dest="batch_size",
    default=8,
    help="Batch size for batched inference, reduce if you run out of memory, "
         "set to 0 for original whisper longform inference",
)

parser.add_argument(
    "--language",
    type=str,
    default="",
    choices=whisper_langs,
    help="Language spoken in the audio, specify None to perform language detection",
)

parser.add_argument(
    "--device",
    dest="device",
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="if you have a GPU use 'cuda', otherwise 'cpu'",
)

parser.add_argument(
    "--nemo-config-file",
    type=str,
    dest="nemo_config_file",
    default=script_dir + "/nemo_msdd_configs/diar_infer_telephonic.yaml",
    help="Set path to config file for NeMo",
)

parser.add_argument(
    "--msdd-model",
    type=str,
    dest="msdd_model",
    default="diar_msdd_telephonic",
    help="Set msdd_model default is diar_msdd_telephonic",
)

parser.add_argument(
    "--use-denoise",
    type=bool,
    dest="is_denoise",
    default=False,
    help="If True use DeepFilterNet (with default script params)",
)

args = parser.parse_args()
language = process_language_arg(args.language, args.model_name)

msdd_model = args.msdd_model
print("Current msdd_model: " + msdd_model)
# if not os.path.isfile(msdd_model):
#     raise FileNotFoundError(f"Provided msdd_model '{msdd_model}' not found check path or permissions.")

path_to_neural_diarizer_cfg = args.nemo_config_file
print("Current config_file: " + path_to_neural_diarizer_cfg)
if not os.path.isfile(path_to_neural_diarizer_cfg):
    raise FileNotFoundError(
        f"Provided msdd config '{path_to_neural_diarizer_cfg}' not found check path or permissions.")

if args.stemming:
    # Isolate vocals from the rest of the audio

    return_code = os.system(
        f'python -m demucs.separate -n htdemucs --two-stems=vocals "{args.audio}" -o temp_outputs --device "{args.device}"'
    )

    if return_code != 0:
        logging.warning(
            "Source splitting failed, using original audio file. "
            "Use --no-stem argument to disable it."
        )
        vocal_target = args.audio
    else:
        vocal_target = os.path.join(
            "temp_outputs",
            "htdemucs",
            os.path.splitext(os.path.basename(args.audio))[0],
            "vocals.wav",
        )
else:
    vocal_target = args.audio

# Apply denoise with DeepFilterNet with some script defaults
if args.is_denoise:
    # TODO: change path or get from args
    default_denoise_model_dir = f"{script_path}/DeepFilterNet/models/DeepFilterNet3/"
    # default_denoise_model = "DeepFilterNet3_ll_onnx"
    default_denoise_model = "model_120.ckpt.best"

    print(f"Initializing DeepFilterNet model: {default_denoise_model} from {default_denoise_model_dir}")

    model, df_state, _ = init_df(model_base_dir=default_denoise_model_dir, default_model=default_denoise_model)

    print(f"Loading audio for denoising from: {vocal_target}")

    try:
        audio_data_to_denoise, original_sr = load_audio(vocal_target, sr=df_state.sr())
        print(f"Audio loaded successfully, sample rate: {original_sr}")  # Перевірка частоти
    except Exception as e:
        logging.error(f"Failed to load audio file {vocal_target} for DeepFilterNet: {e}")

        # Вирішіть, що робити далі - пропустити denoising чи зупинити скрипт
        # Наприклад, просто використовуємо оригінальний файл без denoising:
        # audio_data_to_denoise = None
        raise e  # Або зупиняємо скрипт

    if audio_data_to_denoise is not None:
        print("Applying DeepFilterNet enhancement...")

        # Передаємо завантажені аудіо дані (Tensor або NumPy array)
        enhanced_audio_data = enhance(model, df_state, audio_data_to_denoise)

        print("Enhancement with DeepFilterNet complete.")

        try:
            save_audio(vocal_target, enhanced_audio_data, sr=df_state.sr())
            print(f"Denoised audio saved back to: {vocal_target}")
        except Exception as e:
            logging.error(f"Failed to save enhanced audio file {vocal_target}: {e}")
            raise e  # Або якось інакше обробити помилку збереження
    else:
        logging.warning("Skipping DeepFilterNet enhancement due to audio loading failure.")

    # free memory of usage DeepFilterNet
    print("Cleaning up DeepFilterNet resources...")
    del model
    del df_state
    if 'audio_data_to_denoise' in locals() and audio_data_to_denoise is not None:
        del audio_data_to_denoise
    if 'enhanced_audio_data' in locals() and enhanced_audio_data is not None:
        del enhanced_audio_data
    collected_count = gc.collect()
    torch.cuda.empty_cache()  # Додатково очистити кеш GPU
    print(f"Garbage Collector released DeepFilterNet {collected_count} objects.")

# Transcribe the audio file

whisper_model = faster_whisper.WhisperModel(
    args.model_name, device=args.device, compute_type=mtypes[args.device]
)
whisper_pipeline = faster_whisper.BatchedInferencePipeline(whisper_model)
audio_waveform = faster_whisper.decode_audio(vocal_target)
suppress_tokens = (
    find_numeral_symbol_tokens(whisper_model.hf_tokenizer)
    if args.suppress_numerals
    else [-1]
)

if args.batch_size > 0:
    transcript_segments, info = whisper_pipeline.transcribe(
        audio_waveform,
        language,
        suppress_tokens=suppress_tokens,
        batch_size=args.batch_size,
    )
else:
    transcript_segments, info = whisper_model.transcribe(
        audio_waveform,
        language,
        suppress_tokens=suppress_tokens,
        vad_filter=True,
    )

full_transcript = "".join(segment.text for segment in transcript_segments)

# clear gpu vram
del whisper_model, whisper_pipeline
torch.cuda.empty_cache()

# Forced Alignment
alignment_model, alignment_tokenizer = load_alignment_model(
    args.device,
    dtype=torch.float16 if args.device == "cuda" else torch.float32,
)

emissions, stride = generate_emissions(
    alignment_model,
    torch.from_numpy(audio_waveform)
    .to(alignment_model.dtype)
    .to(alignment_model.device),
    batch_size=args.batch_size,
)

del alignment_model
torch.cuda.empty_cache()

tokens_starred, text_starred = preprocess_text(
    full_transcript,
    romanize=True,
    language=langs_to_iso[info.language],
)

segments, scores, blank_token = get_alignments(
    emissions,
    tokens_starred,
    alignment_tokenizer,
)

spans = get_spans(tokens_starred, segments, blank_token)

word_timestamps = postprocess_results(text_starred, spans, stride, scores)

# convert audio to mono for NeMo combatibility
ROOT = os.getcwd()
temp_path = os.path.join(ROOT, "temp_outputs")
os.makedirs(temp_path, exist_ok=True)
torchaudio.save(
    os.path.join(temp_path, "mono_file.wav"),
    torch.from_numpy(audio_waveform).unsqueeze(0).float(),
    16000,
    channels_first=True,
)

# Initialize NeMo MSDD diarization model
# msdd_model = NeuralDiarizer(cfg=create_config(temp_path)).to(args.device)
# msdd_model = NeuralDiarizer(cfg=create_config_custom(temp_path)).to(args.device)

msdd_model = NeuralDiarizer(cfg=load_config(temp_path, path_to_neural_diarizer_cfg, msdd_model)).to(args.device)
msdd_model.diarize()

del msdd_model
torch.cuda.empty_cache()

# Reading timestamps <> Speaker Labels mapping


speaker_ts = []
with open(os.path.join(temp_path, "pred_rttms", "mono_file.rttm"), "r") as f:
    lines = f.readlines()
    for line in lines:
        line_list = line.split(" ")
        s = int(float(line_list[5]) * 1000)
        e = s + int(float(line_list[8]) * 1000)
        speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])

wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

if info.language in punct_model_langs:
    # restoring punctuation in the transcript to help realign the sentences
    punct_model = PunctuationModel(model="kredor/punctuate-all")

    words_list = list(map(lambda x: x["word"], wsm))

    labled_words = punct_model.predict(words_list, chunk_size=230)

    ending_puncts = ".?!"
    model_puncts = ".,;:!?"

    # We don't want to punctuate U.S.A. with a period. Right?
    is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

    for word_dict, labeled_tuple in zip(wsm, labled_words):
        word = word_dict["word"]
        if (
                word
                and labeled_tuple[1] in ending_puncts
                and (word[-1] not in model_puncts or is_acronym(word))
        ):
            word += labeled_tuple[1]
            if word.endswith(".."):
                word = word.rstrip(".")
            word_dict["word"] = word

else:
    logging.warning(
        f"Punctuation restoration is not available for {info.language} language."
        " Using the original punctuation."
    )

wsm = get_realigned_ws_mapping_with_punctuation(wsm)
ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

with open(f"{os.path.splitext(args.audio)[0]}.txt", "w", encoding="utf-8-sig") as f:
    get_speaker_aware_transcript(ssm, f)

with open(f"{os.path.splitext(args.audio)[0]}.srt", "w", encoding="utf-8-sig") as srt:
    write_srt(ssm, srt)

cleanup(temp_path)
