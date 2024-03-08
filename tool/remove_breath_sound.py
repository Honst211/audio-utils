import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import click


@click.command()
@click.option("--audio_path", type=str)
@click.option("--output_path", type=str)
def remove_breath_sound(audio_path, output_path):
    # 读取音频文件
    y, sr = librosa.load(audio_path)

    # 计算幅度谱
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

    # 设置谱减阈值
    threshold = -30

    # 将低于阈值的谱点设置为0
    D[ D < threshold ] = 0

    # 逆变换得到处理后的音频
    y_processed = librosa.istft(librosa.db_to_amplitude(D) * np.exp(1j * np.angle(librosa.stft(y))))

    # 保存处理后的音频
    librosa.output.write_wav(output_path, y_processed, sr=sr)


if __name__ == "__main__":
    remove_breath_sound()
