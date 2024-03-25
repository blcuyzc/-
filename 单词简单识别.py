from scipy.io import wavfile
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T


labels = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']
new_sample_rate = 8000

transform   = torchaudio.transforms.Resample(orig_freq=16000, new_freq=new_sample_rate)

def label_to_index(word):
    # Return the position of the word in labels
    return torch.tensor(labels.index(word))

def index_to_label(index):
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
    return labels[index]

class M5(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)

def read_wav_file(filepath):
    # 使用 scipy 读取 WAV 文件
    sample_rate, data = wavfile.read(filepath)
    return sample_rate, data
# 找到最有可能的标签
def get_likely_index(tensor):
    return tensor.argmax(dim=-1)

def predict(tensor):
    model = M5()
    model.load_state_dict(torch.load(r'C:\Users\32882\PycharmProjects\torch_li_mu\input\model\audio_model.pth'))
    tensor = transform(tensor)
    tensor = model(tensor.unsqueeze(0))
    tensor = get_likely_index(tensor)
    tensor = index_to_label(tensor.squeeze())
    return tensor


# 用于将从文件读取的数据转换成模型需要的格式的函数
def transform_audio_data(data, sample_rate, new_sample_rate=8000):
    # 如果需要，将音频数据重采样到新的采样率
    if sample_rate != new_sample_rate:
        resample_transform = T.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
        data = resample_transform(data)
    return data

def predict_wav_file(wav_file_path, model):
    # 读取.wav文件
    sample_rate, waveform_data = wavfile.read(wav_file_path)
    waveform_data = torch.tensor(waveform_data, dtype=torch.float32)

    # 如果有多个声道，取平均值合并为单声道
    if waveform_data.ndim > 1:
        waveform_data = torch.mean(waveform_data, dim=1)

    # 将音频数据转换为模型需要的格式
    transformed_data = transform_audio_data(waveform_data, sample_rate)

    # 添加一个维度以表示批次大小 [batch_size, channels, length]
    transformed_data = transformed_data.unsqueeze(0).unsqueeze(0)

    # 使用模型进行预测
    model.eval()  # 将模型设置为评估模式
    with torch.no_grad():  # 关闭梯度计算
        outputs = model(transformed_data)
        # 首先确保outputs是[1, num_classes]形状的张量
        outputs = outputs.squeeze()  # 如果是[1, num_classes]，squeeze后会变成[num_classes]
        _, predicted_index = torch.max(outputs, 0)  # 在0维度上找最大值
        print(predicted_index)  # 应该输出单个整数索引
        predicted_label = labels[predicted_index.item()]  # 使用 .item() 获取tensor中的值
    return predicted_label

# 加载模型
model = M5(n_input=1, n_output=len(labels), stride=16, n_channel=32)
model.load_state_dict(torch.load(r'C:\Users\32882\PycharmProjects\torch_li_mu\input\model\audio_model.pth', map_location=torch.device('cpu')))

# 指定.wav文件路径
wav_file_path = r"C:\Users\32882\PycharmProjects\torch_li_mu\input\audio\wav\20240323_212157.wav"
# 读取 WAV 文件

# 预测.wav文件中的单词
predicted_word = predict_wav_file(wav_file_path, model)

print(f"Predicted word: {predicted_word}")
