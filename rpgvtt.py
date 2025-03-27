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
from whisper import available_models
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
from io import StringIO
import re
import requests

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
        print(f"\n🔊 {audio_path} (press any key to skip)")

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
            print(f"\n❌ Play audio .mp3 completed: {str(e)}")
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

            self.preprocess_save(transcribeResult, mp3_path.with_suffix(".process"))

            print(f"\n🔨 pre-transcribe finished")
            print(f"\n✅ .PROCESS created: {mp3_path.parent}/{mp3_path.name}")
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
            f"\n✅ .{output_format.upper()} created: {mp3_path.parent}/{mp3_path.name}"
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

            print(f"\n✅ .SPK created: {mp3_path.parent}/{mp3_path.name}")

            print(f"\n🔨 Diarization finished")
            print(f"\n{'='*40}")

        except subprocess.CalledProcessError as e:
            print(f"\n❌ Diarization error: {str(e)}")
            print(f"\n{'='*40}")
            return None

    def process_resume(self, mp3_path, transcribeResult):
        """Generate resume with AI API"""

        API_CONFIG = {
            "openai": {
                "url": "https://api.openai.com/v1/chat/completions",
                "key": os.getenv("OPENAI_API_KEY"),
                "model": "gpt-4",
                "headers": lambda key: {"Authorization": f"Bearer {key}"},
            },
            "deepseek": {
                "url": "https://api.deepseek.com/v1/chat/completions",
                "key": os.getenv("DEEPSEEK_API_KEY"),
                "model": "deepseek-chat",
                "headers": lambda key: {"Authorization": f"Bearer {key}"},
            },
        }

        audio_basename = re.sub(
            r"\.mp3$", "", os.path.basename(mp3_path), flags=re.IGNORECASE
        )

        # Corrigir geração do transcript
        buffer = StringIO()
        WriteTXT(mp3_path.parent).write_result(transcribeResult, file=buffer)
        transcript = buffer.getvalue()

        buffer.close()

        prompt = (
            "Crie um resumo jornalístico de uma sessão de RPG baseado na transcrição abaixo.\n"
            "Estrutura requerida:\n"
            "- Primeiro parágrafo: visão geral concisa (3-5 frases) com os fatos principais\n"
            "- Parágrafos seguintes: detalhamento dos eventos mais relevantes (2-4 parágrafos)\n"
            "Características:\n"
            "- Tom formal e imparcial\n"
            "- Estilo similar a reportagem\n"
            "- Destaque para decisões importantes e eventos cruciais\n"
            "- Evite gírias e mantenha coerência temporal\n"
            f"Transcrição:\n{transcript}"
        )

        summary = None
        service = None

        # Tentar serviços em ordem de preferência
        for service_name in ["openai", "deepseek"]:
            config = API_CONFIG[service_name]
            if config["key"]:
                try:
                    headers = {
                        "Content-Type": "application/json",
                        **config["headers"](config["key"]),
                    }

                    payload = {
                        "model": config["model"],
                        "messages": [
                            {
                                "role": "system",
                                "content": "Você é um jornalista experiente resumindo eventos de uma sessão de RPG.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0.7,
                        "max_tokens": 1000,
                    }

                    print("\n=== Debug: Sending Request ===")

                    start_time = time.time()

                    response = requests.post(
                        config["url"], headers=headers, json=payload, timeout=300
                    )

                    elapsed = time.time() - start_time
                    print(f"⏱️ Response time: {elapsed:.2f}s")

                    response.raise_for_status()  # Lança exceção para status 4xx/5xx

                    print(f"✅ {service_name} response: {response.status_code}")
                    summary = response.json()["choices"][0]["message"]["content"]

                    response.raise_for_status()  # Isso vai levantar uma exceção para códigos 4xx/5xx

                    if response.status_code == 200:
                        summary = response.json()["choices"][0]["message"]["content"]
                        service = service_name
                        break
                    else:
                        print(
                            f"Error {service_name} ({response.status_code}): {response.text}"
                        )
                except requests.exceptions.ConnectionError as e:
                    print(f"❌ Connection failed to {service_name}: {str(e)}")
                except requests.exceptions.Timeout:
                    print(f"⏰ Timeout connecting to {service_name} (5s)")
                except requests.exceptions.HTTPError as e:
                    print(f"🚨 HTTP error from {service_name}: {str(e)}")
                except Exception as e:
                    print(f"⚠️ Unexpected error with {service_name}: {str(e)}")
                continue

        if summary:
            output_file = os.path.join(mp3_path.parent, f"{audio_basename}.resume")

            with open(output_file, "w", encoding="utf-8") as file:
                file.write(f"Resume created by {service}:\n\n{summary}")
            print(f"\n✅ .RESUME created: {mp3_path.parent}/{mp3_path.name}")
            return summary

        print("\n❌ No AI service available (check your API Keys)")
        return None

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
def listar_arquivos_mp3(pasta_base: list[str]) -> list[Path]:
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


def report(mp3_files, args):
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

    print("\n📊 Status the Files:")
    print(
        tabulate(
            status,
            headers=["File", "Folder", "Process", "VTT", "SPK", "SRT", "RESUME"],
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
        Generate --process is {"✅" if args.process else "❌"}       ⚙️ process: {process} / {total} {(process/total)*100:.0f}%
        Generate --spk is     {"✅" if args.spk     else "❌"}       📝 VTT: {vtt} / {total} {(vtt/total)*100:.0f}%
        Generate --vtt is     {"✅" if args.vtt     else "❌"}       🔢 SPK: {spk} / {total} {(spk/total)*100:.0f}%
        Generate --srt is     {"✅" if args.srt     else "❌"}       📝 SRT: {srt} / {total} {(srt/total)*100:.0f}%
        Generate --resume is  {"✅" if args.resume  else "❌"}       📝 RESUME: {resume} / {total} {(resume/total)*100:.0f}%
        """
    )

    return total, process, vtt, spk, srt, resume


def getArgs():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Processador de Áudio RPG",
    )
    parser.add_argument(
        "audio", nargs="+", type=str, help="audio file(s) to transcribe or folder path"
    )
    parser.add_argument(
        "--ls",
        action="store_true",  # default=False,
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
        "--process",
        action="store_true",  # default=False,
        help="Generate process file",
    )
    parser.add_argument(
        "--vtt",
        action="store_true",  # default=False,
        help="Generate vtt file",
    )
    parser.add_argument(
        "--spk",
        action="store_true",  # default=False,
        help="Generate spk file by pyannote",
    )
    parser.add_argument(
        "--srt",
        action="store_true",  # default=False,
        help="Generate srt file by pyannote",
    )
    parser.add_argument(
        "--resume",
        action="store_true",  # default=False,
        help="Generate resume file by gpt-4",
    )
    return parser.parse_args()


# endregion


def main():
    args = getArgs()

    mp3_files = listar_arquivos_mp3(args.audio)
    total, process, vtt, spk, srt, resume = report(mp3_files, args)

    if args.ls:
        return

    controller = VttSpkGenerator(
        auth_token=os.getenv("HF_AUTH_TOKEN"),
        language=args.language,
        model=args.model,
        device=args.device,
        threads=args.threads,
        whisper=args.process and (total - process > 0),  # only load whisper if need
        pyannote=args.spk and (total - spk > 0),  # only load pyannote if need
    )

    # create queues
    process_queue = queue.Queue()
    spk_queue = queue.Queue()
    vtt_queue = queue.Queue()
    srt_queue = queue.Queue()
    resume_queue = queue.Queue()

    def spk_worker():
        """ "Start a thread to process spk files"""
        while args.spk:
            try:
                mp3_path, transcribe = spk_queue.queue[0]  # only see the first item
            except IndexError:
                time.sleep(1)
                continue

            if not mp3_path.with_suffix(".spk").exists():
                try:
                    controller.process_spk(mp3_path, transcribe)
                    # only remove success
                    spk_queue.get()
                    spk_queue.task_done()
                except Exception as e:
                    print(f"Error to process {mp3_path.name}, trying again: {str(e)}")
                    time.sleep(5)
            else:
                spk_queue.get()
                spk_queue.task_done()

    thread_spk = Thread(target=spk_worker, daemon=True)
    thread_spk.start()

    if args.spk:
        notprocessed = []
        # loop for get .process files ready to process spk
        for mp3_path in mp3_files:
            if (
                mp3_path.with_suffix(".process").exists()
                and not mp3_path.with_suffix(".spk").exists()
            ):
                transcribeResult = controller.preprocess_load(
                    mp3_path.with_suffix(".process")
                )
                spk_queue.put([mp3_path, transcribeResult])
            else:
                notprocessed.append(mp3_path)
        if args.process and notprocessed.count > 0:
            print(
                "\n🔍 Need process {notprocessed.count} files for use spk generate..."
            )
    else:
        thread_spk.join()

    def vtt_worker():
        """ "Start a thread to process vtt files"""
        while args.vtt:
            try:
                mp3_path, transcribe = vtt_queue.queue[0]  # only see the first item
            except IndexError:
                time.sleep(1)
                continue

            if not mp3_path.with_suffix(".vtt").exists():
                try:
                    controller.process_whisper_format(mp3_path, transcribe, "vtt")
                    # only remove success
                    vtt_queue.get()
                    vtt_queue.task_done()
                except Exception as e:
                    print(f"Error to process {mp3_path.name}, trying again: {str(e)}")
                    time.sleep(5)
            else:
                vtt_queue.get()
                vtt_queue.task_done()

    thread_vtt = Thread(target=vtt_worker, daemon=True)
    thread_vtt.start()

    if args.vtt:
        notprocessed = []
        # loop for get .process files ready to process vtt
        for mp3_path in mp3_files:
            if (
                mp3_path.with_suffix(".process").exists()
                and not mp3_path.with_suffix(".vtt").exists()
            ):
                transcribeResult = controller.preprocess_load(
                    mp3_path.with_suffix(".process")
                )
                vtt_queue.put([mp3_path, transcribeResult])
            else:
                notprocessed.append(mp3_path)
        if args.process and notprocessed.count > 0:
            print(
                "\n🔍 Need process {notprocessed.count} files for use vtt generate..."
            )
    else:
        thread_vtt.join()

    def srt_worker():
        """ "Start a thread to process srt files"""
        while args.srt:
            try:
                mp3_path, transcribe = srt_queue.queue[0]  # only see the first item
            except IndexError:
                time.sleep(1)
                continue

            if not mp3_path.with_suffix(".srt").exists():
                try:
                    controller.process_whisper_format(mp3_path, transcribe, "srt")
                    # only remove success
                    srt_queue.get()
                    srt_queue.task_done()
                except Exception as e:
                    print(f"Error to process {mp3_path.name}, trying again: {str(e)}")
                    time.sleep(5)
            else:
                # Já existe, remove da fila
                srt_queue.get()
                srt_queue.task_done()

    thread_srt = Thread(target=srt_worker, daemon=True)
    thread_srt.start()

    if args.srt:
        notprocessed = []
        # loop for get .process files ready to process srt
        for mp3_path in mp3_files:
            if (
                mp3_path.with_suffix(".process").exists()
                and not mp3_path.with_suffix(".srt").exists()
            ):
                transcribeResult = controller.preprocess_load(
                    mp3_path.with_suffix(".process")
                )
                srt_queue.put([mp3_path, transcribeResult])
            else:
                notprocessed.append(mp3_path)
        if args.process and notprocessed.count > 0:
            print(
                "\n🔍 Need process {notprocessed.count} files for use srt generate..."
            )
    else:
        thread_srt.join()

    def resume_worker():
        """ "Start a thread to process resume files"""
        while args.resume:

            try:
                mp3_path, transcribe = resume_queue.queue[0] # only see the first item
            except IndexError:
                time.sleep(1)
                continue

            if not mp3_path.with_suffix(".resume").exists():
                try:
                    controller.process_resume(mp3_path, transcribe)
                    # only remove success
                    resume_queue.get()
                    resume_queue.task_done()
                except Exception as e:
                    print(f"Error to process {mp3_path.name}, trying again: {str(e)}")
                    time.sleep(5)
            else:
                resume_queue.get()
                resume_queue.task_done()

    thread_resume = Thread(target=resume_worker, daemon=True)
    thread_resume.start()

    if args.resume:
        notprocessed = []
        for mp3_path in mp3_files:
            if (
                mp3_path.with_suffix(".process").exists()
                and not mp3_path.with_suffix(".resume").exists()
            ):
                transcribeResult = controller.preprocess_load(
                    mp3_path.with_suffix(".process")
                )
                resume_queue.put([mp3_path, transcribeResult])
            else:
                notprocessed.append(mp3_path)
        if args.process and notprocessed.count > 0:
            print("\n🔍 Need process {notprocessed.count} files for use resume")
    else:
        thread_resume.join()

    def process_worker():
        """ "Start a thread to process process files"""

        while args.process:
            # Verifica o próximo item sem removê-lo
            try:
                mp3_path = process_queue.queue[0]  # Apenas olha o primeiro item
            except IndexError:
                time.sleep(1)
                continue

            if not mp3_path.with_suffix(".process").exists():
                print(f"\n🔨 Starting process... {mp3_path}")
                try:
                    transcribeResult = controller.preprocess(mp3_path)
                    controller.play_30(mp3_path)

                    if args.spk:
                        if not mp3_path.with_suffix(".spk").exists():
                            spk_queue.put([mp3_path, transcribeResult])

                    if args.vtt:
                        if not mp3_path.with_suffix(".vtt").exists():
                            vtt_queue.put([mp3_path, transcribeResult])

                    if args.srt:
                        if not mp3_path.with_suffix(".srt").exists():
                            srt_queue.put([mp3_path, transcribeResult])

                    if args.resume:
                        if not mp3_path.with_suffix(".resume").exists():
                            resume_queue.put([mp3_path, transcribeResult])

                    # Só remove se processou com sucesso
                    process_queue.get()
                    process_queue.task_done()
                except Exception as e:
                    print(f"Error to process {mp3_path.name}, trying again: {str(e)}")
                    time.sleep(5)
            else:
                # Já existe, remove da fila
                process_queue.get()
                process_queue.task_done()

    thread_process = Thread(target=process_worker, daemon=True)
    thread_process.start()

    if args.process:
        for mp3_path in mp3_files:
            if not mp3_path.with_suffix(".process").exists():
                process_queue.put(mp3_path)
                thread_process.join()  # force 1 by 1 processing, test your machine before remove this line
    else:
        thread_process.join()

    # finish all threads
    print("\n🔚 All threads don't have work?")
    # if a all queue is empty the thread is finished
    while not (
        process_queue.empty()
        and spk_queue.empty()
        and vtt_queue.empty()
        and srt_queue.empty()
        and resume_queue.empty()
    ):
        time.sleep(1)
    report(mp3_files, args)
    print("\n🔚 Work Done")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n� Skipped by user")
