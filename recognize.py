import argparse
import os

from audio_service import AudioService


def get_args():
    parser = argparse.ArgumentParser(description="Audio Recognition",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--wav_path", help="path to wav audio", required=True)
    return parser.parse_args()


def main():
    args = get_args()
    original_wav_file = args.wav_path

    if os.path.exists(original_wav_file):
        audio_service = AudioService()
        wav_file = audio_service.convert_audio(original_wav_file)
        recognized_text = audio_service.recognize(wav_file)
        print("Recognized Text:", recognized_text)


main()
