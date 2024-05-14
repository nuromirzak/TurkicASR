import subprocess

import wave
import numpy as np
from espnet2.bin.asr_inference import Speech2Text


class AudioService:
    def __init__(self):
        # cuda_available = torch.cuda.is_available()
        cuda_available = True
        print(f"CUDA available: {cuda_available}")

        self.asr_model_path = "exp/asr_train_asr_1410_raw_all_turkic_1610_char_sp"
        self.lm_model_path = "exp/lm_train_lm_1410_all_turkic_1610_char"
        self.train_config = self.asr_model_path + "/config.yaml"
        self.model_file = self.asr_model_path + "/valid.acc.ave_10best.pth"
        self.lm_config = self.lm_model_path + "/config.yaml"
        self.lm_file = self.lm_model_path + "/valid.loss.ave_10best.pth"
        self.speech2text = Speech2Text(
            asr_train_config=self.train_config,
            asr_model_file=self.model_file,
            lm_train_config=self.lm_config,
            lm_file=self.lm_file,
            token_type=None,
            bpemodel=None,
            maxlenratio=0.0,
            minlenratio=0.0,
            beam_size=10,
            ctc_weight=0.5,
            lm_weight=0.3,
            penalty=0.0,
            nbest=1,
            device="cuda" if cuda_available else "cpu"
        )

    def convert_audio(self, input_path):
        print("convert_audio", input_path)
        output_path = input_path.rsplit('.', 1)[0] + "_16k.wav"
        print("output_path", output_path)
        command = ["ffmpeg", "-i", input_path, "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000", output_path]
        subprocess.run(command)
        print("subprocess.run(command) finished")
        return output_path

    def recognize(self, wavfile_path):
        print("recognize", wavfile_path)
        with wave.open(wavfile_path, 'rb') as wavfile:
            buf = wavfile.readframes(-1)
            data = np.frombuffer(buf, dtype='int16')
        speech = data.astype(np.float16) / 32767.0
        print("speech")
        results = self.speech2text(speech)
        print("results")
        return results[0][0]
