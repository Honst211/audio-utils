import click
import pywt
import numpy as np
import scipy.io.wavfile as wav


def thresholding(signal, threshold):
    # 小波变换
    coeffs = pywt.wavedec(signal, 'db4', level=6)

    # 对各层系数进行阈值处理
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold, mode='soft')

    # 逆小波变换
    denoised_signal = pywt.waverec(coeffs, 'db4')

    return denoised_signal


@click.command()
@click.option("--audio_path", type=str)
@click.option("--output_path", type=str)
def main(audio_path, output_path):
    # 读取音频文件
    sample_rate, signal = wav.read(audio_path)

    # 将信号转为浮点数
    signal = np.float32(signal)

    # 设置阈值，可以根据实际情况调整
    threshold = 1

    # 应用小波阈值降噪处理
    denoised_signal = thresholding(signal, threshold)

    # 将处理后的信号保存到新的文件
    wav.write(output_path, sample_rate, np.int16(denoised_signal))


if __name__ == "__main__":
    main()
