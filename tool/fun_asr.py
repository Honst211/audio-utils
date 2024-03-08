import functools
import threading

import click
from pathlib import Path

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from tqdm import tqdm


threads = []

@functools.lru_cache(maxsize=1)
def asr_pipeline():
    return pipeline(
        task=Tasks.auto_speech_recognition,
        model='damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
    )


def transcribe(audio_file):
    _pipeline = asr_pipeline()
    rec_result = _pipeline(audio_in=audio_file)
    print(rec_result["text"])
    return rec_result["text"]


@click.command()
@click.option("--audio_dir", type=str)
def main(audio_dir):
    audio_dir = Path(audio_dir)

    wav_files = list(audio_dir.glob("*.wav"))
    flac_files = list(audio_dir.glob("*.flac"))
    mp3_files = list(audio_dir.glob("*.mp3"))

    all_audio_files = wav_files + flac_files + mp3_files

    for filepath in tqdm(all_audio_files, desc="Processing files"):
        thread = threading.Thread(target=transcribe, args=(str(filepath),))
        threads.append(thread)

    # 启动所有线程
    for thread in threads:
        thread.start()

    # 等待所有线程执行完毕，并获取结果
    results = []

    for thread in threads:
        thread.join()
        result = thread._target(*thread._args, **thread._kwargs)  # 手动执行线程的目标函数，获取返回值
        results.append(result)

    # 输出结果
    for i, result in enumerate(results):
        for text in result:
            with open((audio_dir / filepath.stem).with_suffix(".lab"), "w", encoding="utf-8") as f:
                f.write(text)


if __name__ == "__main__":
    main()
