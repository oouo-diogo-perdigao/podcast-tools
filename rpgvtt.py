import os
import subprocess
import time
import queue
import argparse
import numpy as np
import torch
import pickle
import re
from pathlib import Path
from tabulate import tabulate
from datetime import datetime
from pynput import keyboard
from typing import Literal, cast
from dotenv import load_dotenv
from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE
from whisper.transcribe import transcribe
from whisper.utils import (
    WriteSRT,
    WriteTXT,
    WriteVTT,
    optional_int,
    str2bool,
)
from threading import Thread
from pyannote.core import Segment, Annotation, Timeline

# load environment variables
load_dotenv()


# region (collapsed) minha classe de controle
class VttSpkGenerator:
    def __init__(
        self,
        auth_token,
        language=None,
        model="medium",
        device="cuda",
        threads=0,
        *,
        whisper=True,
        pyannote=True,
    ):
        # uso interno
        self.skip = False
        self.listener = None
        self.process = None

        # parametros de pacotes
        self.language = language

        if threads > 0:
            torch.set_num_threads(threads)

        if whisper:
            thread_whisper = Thread(target=self.load_whisper, args=(model, device))
            thread_whisper.start()
        if pyannote:
            thread_pyannote = Thread(target=self.load_pyannote, args=(auth_token,))
            thread_pyannote.start()

        if whisper:
            thread_whisper.join()
        if pyannote:
            thread_pyannote.join()

    def load_whisper(self, model, device):
        from whisper import load_model

        self.whisper = load_model(model, device)

    def load_pyannote(self, auth_token):
        from pyannote.audio import Pipeline

        self.pyannote = Pipeline.from_pretrained(
            "pyannote/speaker-diarization", use_auth_token=auth_token
        )

    def play_30(self, audio_path):
        print("\n🔊 {audio_path} (pressione qualquer tecla para pular)")

        def on_press(self, key):
            self.skip = True

        self.skip = False
        try:
            # Inicia a reprodução
            self.process = subprocess.Popen(
                ["ffplay", "-nodisp", "-autoexit", "-t", "30", str(audio_path)]
            )

            # Configura o listener de teclado
            self.listener = keyboard.Listener(on_press=on_press)
            self.listener.start()

            # Aguarda término ou interrupção
            start_time = time.time()
            while time.time() - start_time < 30:
                if self.skip:
                    break
                time.sleep(0.1)

            # Finaliza a reprodução
            if self.process.poll() is None:
                self.process.terminate()
                self.process.wait()

        except Exception as e:
            print(f"\n❌ Erro ao reproduzir o áudio: {str(e)}")
        finally:
            if self.listener:
                self.listener.stop()
                self.listener = None

    def preprocess(self, mp3_path):
        if not self.whisper:
            raise ValueError("whisper not loaded")

        print(f"\n{'='*40}")
        print(f"\n🔨 Starting pre-transcribe...")
        print(f"📂 Folder: {mp3_path.parent}")
        print(f"🎧 Name: {mp3_path.name}")
        print(f"⏰ LastModify: {datetime.fromtimestamp(os.path.getmtime(mp3_path))}")
        print(f"{'.'*3}")

        try:
            # cria a pasta de saída se não existir
            os.makedirs(mp3_path.parent, exist_ok=True)

            transcribeResult = transcribe(
                self.whisper,
                str(mp3_path),
                verbose=True,
                # initial_prompt=None,
                task="transcribe",
                language=self.language,
            )

            print(f"\n🔨 pre-transcribe finished")
            print(f"\n{'='*40}")

            return transcribeResult

        except subprocess.CalledProcessError as e:
            print(f"\n❌ pre-transcribe error: {str(e)}")
            print(f"\n{'='*40}")
            return None

    def process_whisper_format(
        self,
        mp3_path,
        transcribeResult,
        output_format: Literal["txt", "vtt", "srt"] = "vtt",
    ):
        audio_basename = re.sub(
            r"\.mp3$", "", os.path.basename(mp3_path), flags=re.IGNORECASE
        )
        if output_format == "txt":
            # save TXT
            with open(
                os.path.join(mp3_path.parent, audio_basename + ".txt"),
                "w",
                encoding="utf-8",
            ) as file:
                WriteTXT(mp3_path.parent).write_result(transcribeResult, file=file)
        elif output_format == "vtt":
            # save VTT
            with open(
                os.path.join(mp3_path.parent, audio_basename + ".vtt"),
                "w",
                encoding="utf-8",
            ) as file:
                WriteVTT(mp3_path.parent).write_result(transcribeResult, file=file)
        elif output_format == "srt":
            # save SRT
            with open(
                os.path.join(mp3_path.parent, audio_basename + ".srt"),
                "w",
                encoding="utf-8",
            ) as file:
                WriteSRT(mp3_path.parent).write_result(transcribeResult, file=file)

        print(
            f"\n✅ {output_format.upper()} created: {mp3_path.parent}/{mp3_path.name}"
        )

    def process_spk(self, mp3_path, transcribeResult):
        try:
            if not self.pyannote:
                raise ValueError("pyannote not loaded")

            print(f"\n{'='*40}")
            print(f"\n🔨 Starting Diarization... .spk")
            print(f"📂 Folder: {mp3_path.parent}")
            print(f"🎧 Name: {mp3_path.name}")
            print(
                f"⏰ LastModify: {datetime.fromtimestamp(os.path.getmtime(mp3_path))}"
            )
            print(f"{'.'*3}")

            audio_basename = os.path.basename(mp3_path)
            diarization_result = self.pyannote(mp3_path)

            timestamp_texts = self.get_text_with_timestamp(transcribeResult)
            spk_text = self.add_speaker_info_to_text(
                timestamp_texts, diarization_result
            )
            res = self.merge_sentence(spk_text)

            with open(
                os.path.join(mp3_path.parent, audio_basename + ".spk"),
                "w",
                encoding="utf-8",
            ) as file:
                for seg, spk, sentence in res:
                    line = f"{seg.start:.2f} {seg.end:.2f} {spk} {sentence}\n"
                    file.write(line)

            print(f"\n🔨 Diarization finished")
            print(f"\n{'='*40}")

        except subprocess.CalledProcessError as e:
            print(f"\n❌ Diarization error: {str(e)}")
            print(f"\n{'='*40}")
            return None

    def process_resume(self, mp3_path, transcribeResult):
        print("not yet implemented")

    def preprocess_save(self, obj, file_path):
        """Save Python object in .process file using pickle."""
        with open(file_path, "wb") as file:
            pickle.dump(obj, file)

    def preprocess_load(self, file_path):
        """Load Python object from .process file using pickle."""
        with open(file_path, "rb") as file:
            return pickle.load(file)

    def get_text_with_timestamp(self, transcribe_res):
        timestamp_texts = []
        for item in transcribe_res["segments"]:
            start = item["start"]
            end = item["end"]
            text = item["text"]
            timestamp_texts.append((Segment(start, end), text))
        return timestamp_texts

    def add_speaker_info_to_text(self, timestamp_texts, ann):
        spk_text = []
        for seg, text in timestamp_texts:
            spk = ann.crop(seg).argmax()
            spk_text.append((seg, spk, text))
        return spk_text

    def merge_cache(self, text_cache):
        sentence = "".join([item[-1] for item in text_cache])
        spk = text_cache[0][1]
        start = text_cache[0][0].start
        end = text_cache[-1][0].end
        return Segment(start, end), spk, sentence

    def merge_sentence(self, spk_text):
        PUNC_SENT_END = [".", "?", "!"]

        merged_spk_text = []
        pre_spk = None
        text_cache = []
        for seg, spk, text in spk_text:
            if spk != pre_spk and pre_spk is not None and len(text_cache) > 0:
                merged_spk_text.append(self.merge_cache(text_cache))
                text_cache = [(seg, spk, text)]
                pre_spk = spk

            elif text and len(text) > 0 and text[-1] in PUNC_SENT_END:
                text_cache.append((seg, spk, text))
                merged_spk_text.append(self.merge_cache(text_cache))
                text_cache = []
                pre_spk = spk
            else:
                text_cache.append((seg, spk, text))
                pre_spk = spk
        if len(text_cache) > 0:
            merged_spk_text.append(self.merge_cache(text_cache))
        return merged_spk_text


# endregion


# region (collapsed) Funções auxiliares locais
def listar_arquivos_mp3(pasta_base):
    mp3_files = []

    for item in pasta_base:
        path = Path(item)
        if path.is_file() and path.suffix.lower() == ".mp3":
            mp3_files.append(path)
        elif path.is_dir():
            for root, _, files in os.walk(path):
                for file in files:
                    if file.lower().endswith(".mp3"):
                        mp3_path = Path(root) / file
                        mp3_files.append(mp3_path)
        else:
            # Handle case where the path does not exist
            print(f"Warning: '{item}' does not exist and will be skipped.")

    # Remove duplicates and sort Z-A
    mp3_files = list(set(mp3_files))
    mp3_files.sort(key=lambda x: x.name.lower(), reverse=True)

    if not mp3_files:
        raise ValueError("Nenhum arquivo .mp3 encontrado")
    return mp3_files


def contar(mp3_files):
    status = []
    for mp3 in mp3_files:
        # verifica se existe um arquivo .vtt com o nome do mp3 na mesma pasta
        process = mp3.with_name(f"{mp3.stem}.process").exists()
        vtt = mp3.with_suffix(".vtt").exists()
        spk = mp3.with_suffix(".spk").exists()
        srt = mp3.with_suffix(".srt").exists()
        resume = mp3.with_suffix(".resume").exists()
        status.append(
            [
                mp3.name,
                str(mp3.parent),
                "✅" if process else "❌",
                "✅" if vtt else "❌",
                "✅" if spk else "❌",
                "✅" if srt else "❌",
                "✅" if resume else "❌",
            ]
        )

    print("\n📊 Status dos Arquivos:")
    print(
        tabulate(
            status,
            headers=["Arquivo", "Pasta", "Process", "VTT", "SPK", "SRT", "RESUME"],
            tablefmt="grid",
            showindex=True,
        )
    )
    total = len(status)
    process = sum(1 for item in status if item[2] == "✅")
    vtt = sum(1 for item in status if item[3] == "✅")
    spk = sum(1 for item in status if item[4] == "✅")
    srt = sum(1 for item in status if item[5] == "✅")
    resume = sum(1 for item in status if item[6] == "✅")

    print(
        f"""
        ⚙️ process: {process} / {total} {(process/total)*100:.0f}%
        📝 VTT: {vtt} / {total} {(vtt/total)*100:.0f}%
        🔢 SPK: {spk} / {total} {(spk/total)*100:.0f}%
        📝 SRT: {srt} / {total} {(srt/total)*100:.0f}%
        📝 RESUME: {resume} / {total} {(resume/total)*100:.0f}%
        """
    )

    return total, vtt, spk, srt, resume


def getArgs():
    from whisper import available_models

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Processador de Áudio RPG",
    )
    parser.add_argument(
        "audio", nargs="+", type=str, help="audio file(s) to transcribe or folder path"
    )
    parser.add_argument(
        "--ls",
        default=False,
        help="Only list files and process",
    )
    parser.add_argument(
        "--model",
        default="medium",
        choices=available_models(),
        help="name of the Whisper model to use",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="device to use for PyTorch inference",
    )
    parser.add_argument(
        "--threads",
        type=optional_int,
        default=0,
        help="number of threads used by torch for CPU inference; supercedes MKL_NUM_THREADS/OMP_NUM_THREADS",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="transcribe",
        choices=["transcribe", "translate"],
        help="whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="pt",
        choices=sorted(LANGUAGES.keys())
        + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]),
        help="language spoken in the audio, specify None to perform language detection",
    )
    parser.add_argument(
        "--vtt",
        type=str2bool,
        default=False,
        help="Generate vtt file",
    )
    parser.add_argument(
        "--srt",
        type=str2bool,
        default=False,
        help="Generate srt file by pyannote",
    )
    parser.add_argument(
        "--spk",
        type=str2bool,
        default=False,
        help="Generate spk file by pyannote",
    )
    parser.add_argument(
        "--resume",
        type=str2bool,
        default=False,
        help="Generate resume file by gpt-4",
    )
    return parser.parse_args()


# endregion


def main():
    args = getArgs()

    mp3_files = listar_arquivos_mp3(args.audio)
    contar(mp3_files)
    if args.ls:
        return

    controller = VttSpkGenerator(
        auth_token=os.getenv("HF_AUTH_TOKEN"),
        language=args.language,
        model=args.model,
        device=args.device,
        threads=args.threads,
    )

    def spk_worker():
        """ "Iniciar um worker para vigiar uma fila, que processara .vtt inseridos nela"""
        while True:
            mp3, transcribe = spk_queue.get()
            if not mp3.with_suffix(".spk").exists():
                controller.process_spk(mp3, transcribe)
            spk_queue.task_done()

    thread_spk = Thread(target=spk_worker, daemon=True)
    thread_spk.start()

    spk_queue = queue.Queue()  # fila de diarização

    if args.spk:
        # primeiro percorre os arquivos .txt salvos de outras iterações e adiciona eles na fila de diarização
        for mp3_path in mp3_files:
            if (
                mp3_path.with_suffix(".process").exists()
                and not mp3_path.with_suffix(".spk").exists()
            ):
                transcribeResult = controller.preprocess_load(
                    mp3_path.with_suffix(".process")
                )
                spk_queue.put([mp3_path, transcribeResult])

    def vtt_worker():
        while True:
            mp3, transcribe = vtt_queue.get()
            if not mp3.with_suffix(".vtt").exists():
                controller.process_whisper_format(mp3, transcribe, "vtt")
            vtt_queue.task_done()

    thread_vtt = Thread(target=vtt_worker, daemon=True)
    thread_vtt.start()

    vtt_queue = queue.Queue()

    if args.vtt:
        # primeiro percorre os arquivos .txt salvos de outras iterações e adiciona eles na fila de diarização
        for mp3_path in mp3_files:
            if (
                mp3_path.with_suffix(".process").exists()
                and not mp3_path.with_suffix(".vtt").exists()
            ):
                transcribeResult = controller.preprocess_load(
                    mp3_path.with_suffix(".process")
                )
                vtt_queue.put([mp3_path, transcribeResult])

    def srt_worker():
        while True:
            mp3, transcribe = srt_queue.get()
            if not mp3.with_suffix(".srt").exists():
                controller.process_whisper_format(mp3, transcribe, "srt")
            srt_queue.task_done()

    thread_srt = Thread(target=srt_worker, daemon=True)
    thread_srt.start()

    srt_queue = queue.Queue()

    if args.srt:
        # primeiro percorre os arquivos .txt salvos de outras iterações e adiciona eles na fila de diarização
        for mp3_path in mp3_files:
            if (
                mp3_path.with_suffix(".process").exists()
                and not mp3_path.with_suffix(".srt").exists()
            ):
                transcribeResult = controller.preprocess_load(
                    mp3_path.with_suffix(".process")
                )
                srt_queue.put([mp3_path, transcribeResult])

    def resume_worker():
        while True:
            mp3, transcribe = resume_queue.get()
            if not mp3.with_suffix(".resume").exists():
                controller.process_resume(mp3, transcribe)
            resume_queue.task_done()

    thread_resume = Thread(target=resume_worker, daemon=True)
    thread_resume.start()

    resume_queue = queue.Queue()

    if args.resume:
        # primeiro percorre os arquivos .txt salvos de outras iterações e adiciona eles na fila de diarização
        for mp3_path in mp3_files:
            if (
                mp3_path.with_suffix(".process").exists()
                and not mp3_path.with_suffix(".resume").exists()
            ):
                transcribeResult = controller.preprocess_load(
                    mp3_path.with_suffix(".process")
                )
                resume_queue.put([mp3_path, transcribeResult])

    # create .process for mp3 files that do not have
    pendentes = [f for f in mp3_files if not f.with_suffix(".process").exists()]
    for idx, mp3_path in enumerate(pendentes, 1):
        print(f"📁 File {idx}/{len(pendentes)}")

        transcribeResult = controller.preprocess(mp3_path)
        controller.preprocess_save(transcribeResult, mp3_path.with_suffix(".process"))
        controller.play_30(mp3_path)

        if idx < len(pendentes):
            ("Next file")

        if args.spk:
            if mp3_path.with_suffix(".vtt").exists():
                spk_queue.put([mp3_path, transcribeResult])

        if args.vtt:
            if mp3_path.with_suffix(".process").exists():
                spk_queue.put([mp3_path, transcribeResult])

        if args.srt:
            if mp3_path.with_suffix(".process").exists():
                srt_queue.put([mp3_path, transcribeResult])

        if args.resume:
            if mp3_path.with_suffix(".process").exists():
                resume_queue.put([mp3_path, transcribeResult])

    # Aguarda a conclusão da thread_spk antes de prosseguir
    thread_spk.join()
    thread_vtt.join()
    thread_srt.join()
    thread_resume.join()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Execução interrompida pelo usuário!")
