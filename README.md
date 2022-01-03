# KhehUe-HapSing
自 https://github.com/fatchord/WaveRNN 來訓練。

## 資料
1. 羅馬字：`Kiung ha loiˇ liau dong senˊ qi`
2. Mel Spectrogram頻譜
3. 音檔

## 模型
### Tacotron模型
`1. 羅馬字`轉`2. Mel Spectrogram頻譜`

### Griffinlim數學方法
`2. Mel Spectrogram頻譜`轉`3. 音檔`。轉較遽，像電子聲。

### WaveRNN模型
`2. Mel Spectrogram頻譜`轉`3. 音檔`。愛GPU算1~3秒，像人聲。

## 安
- [Nvidia GPU Driver](https://phoenixnap.com/kb/install-nvidia-drivers-ubuntu)
- [dobi](https://github.com/dnephin/dobi)
- [docker](https://docs.docker.com/engine/installation/linux/docker-ce/ubuntu/)
- [docker-compose](https://docs.docker.com/compose/install/)
- 設定docker權限`sudo usermod -aG docker $USER`

## 訓練步
1. 先用`time dobi hazoi-ngiliau`，會下載客語教典詞音檔，下載好會恁樣：
```
1-ciidien-20190516/
├── corpus
│   ├── s_sound
│   │   ├── 00001.mp3
│   │   ├── 00002.mp3
│   │   ├── ...
│   │   └── 15486.mp3
│   ├── s_sound2
│   │   └── ...
│   ├── s_sound3
│   │   └── ...
│   ├── s_sound4
│   │   └── ...
│   ├── s_sound5
│   │   └── ...
│   └── s_sound6
│        └── ...
└── moe-hakkadict-main
    ├── README.md
    ├── 調值資料_raw
    │   ├── 《臺灣客家語常用詞辭典》內容資料(1100430).csv
    │   ├── 《臺灣客家語常用詞辭典》內容資料(1100430).ods
    │   └── 《臺灣客家語常用詞辭典》內容資料(1100430).pdf
    ├── 調值資料_uni
    │   └── 《臺灣客家語常用詞辭典》內容資料(1100430).csv
    ├── 調型資料
    │   └── 《臺灣客家語常用詞辭典》內容資料(1100430).csv
    └── 轉做調型資料.py
```
2. `time dobi zon-bienma`，毋愛頭尾無聲个部份，tacotron較會收斂，而且wave downsample 乜降做 16bits wav，盡尾合成較遽。程式愛走半點鐘。
```
2-ciidien-20190516-16k/
├── corpus
│   ├── s_sound
│   │   ├── 00001.mp3.wav
│   │   ├── 00002.mp3.wav
│   │   ├── ...
│   │   └── 15486.mp3.wav
│   ├── s_sound2
│   │   └── ...
│   ├── s_sound3
│   │   └── ...
│   ├── s_sound4
│   │   └── ...
│   ├── s_sound5
│   │   └── ...
│   └── s_sound6
│        └── ...
└── moe-hakkadict-main
    ├── README.md
    ├── 調值資料_raw
    │   ├── 《臺灣客家語常用詞辭典》內容資料(1100430).csv
    │   ├── 《臺灣客家語常用詞辭典》內容資料(1100430).ods
    │   └── 《臺灣客家語常用詞辭典》內容資料(1100430).pdf
    ├── 調值資料_uni
    │   └── 《臺灣客家語常用詞辭典》內容資料(1100430).csv
    ├── 調型資料
    │   └── 《臺灣客家語常用詞辭典》內容資料(1100430).csv
    └── 轉做調型資料.py
```
3. `NGINGIEN=SinZhug time dobi preprocess-tacotron`，準備tactorn格式，產生音檔長短`dataset.pkl`，音檔羅馬字對應`text_dict.pkl`、音檔頻譜`mel/`、音檔波形sample`quant/`。
```
3-ciidien-20190516-16k-MeuLid-data/
├── dataset.pkl
├── gta
├── mel
│   ├── 00001.mp3.npy
│   ├── 00002.mp3.npy
│   ├── ...
│   └── 15450.mp3.npy
├── quant
│   ├── 00001.mp3.npy
│   ├── 00002.mp3.npy
│   ├── ...
│   └── 15450.mp3.npy
└── text_dict.pkl
```
4. `NGINGIEN=SinZhug time dobi tacotron`，訓練Tacotron模型。盡尾會產生`gta/`檔案。`gta/`係[Ground Truth Aligned synthesis](https://github.com/Rayhane-mamah/Tacotron-2#synthesis)用个，[Ground Truth相關資料](https://www.aptiv.com/en/insights/article/what-is-ground-truth)。程式愛走12點鐘。
```
4-ciidien-20190516-16k-MeuLid-checkpoints/
├── hagfa_lsa_smooth_attention.tacotron
│   ├── attention
│   │   ├── 100463.png
│   │   ├── 101429.png
│   │   └── ...
│   ├── latest_optim.pyt
│   ├── latest_weights.pyt
│   ├── log.txt
│   ├── mel_plots
│   │   ├── 100463.png
│   │   ├── 101429.png
│   │   └── ...
│   ├── taco_step100K_optim.pyt
│   ├── taco_step100K_weights.pyt
│   ├── ...
│   ├── taco_step98K_optim.pyt
│   └── taco_step98K_weights.pyt
└── hagfa_raw.wavernn
```

- 10000steps个時節愛有線，代表tacotron學着對應`羅馬字`摎`mel/`。假使無，請檢查`text_dict.pkl`對應有著無。
  - 成功个（`4-ciidien-20190516-16k-MeuLid-checkpoints/hagfa_lsa_smooth_attention.tacotron/attention/9659.png`）
  ![成功个attention](tu/siingung-9659.png)
  - 失敗个（`4-ciidien-20190516-16k-MeuLid-checkpoints/hagfa_lsa_smooth_attention.tacotron/attention/9658.png`）
  ![失敗个attention](tu/siidpai-9658.png)

4-1. 假使在tactorn訓練時節，愛產生`gta/`檔案，走`NGINGIEN=SinZhug time dobi tacotron-gta`

4-2. `time dobi habsang-griffinlim`，試合成語句。`griffinlim`个音檔有電子聲，故所愛訓練`WaveRNN`分聲像人講話。
```
5-ciidien-20190516-16k-MeuLid-model_outputs/
├── hagfa_lsa_smooth_attention.tacotron
│   └── __input_Kiung ha l_griffinlim_350k.wav
└ ...
```
5. `NGINGIEN=SinZhug time dobi preprocess-wavernn`，因為太長个音檔無法度用訓練tacotron，會無法度coverage。`hparams.py`有設定`tts_max_mel_len`，故所`gta/`無全部音檔有。這指令照`gta/`檔案，產生wavernn愛个`dataset_wavernn.pkl`。
```
3-ciidien-20190516-16k-MeuLid-data/
├── dataset.pkl
├── dataset_wavernn.pkl
└── ...
```
另外有用`find -delete`㓾0.18秒以下个音檔。
```
Original Traceback (most recent call last):
  File "/opt/conda/lib/python3.6/site-packages/torch/utils/data/_utils/worker.py", line 178, in _worker_loop
    data = fetcher.fetch(index)
  File "/opt/conda/lib/python3.6/site-packages/torch/utils/data/_utils/fetch.py", line 47, in fetch
    return self.collate_fn(data)
  File "/WaveRNN/utils/dataset.py", line 70, in collate_vocoder
    mel_offsets = [np.random.randint(0, offset) for offset in max_offsets]
  File "/WaveRNN/utils/dataset.py", line 70, in <listcomp>
    mel_offsets = [np.random.randint(0, offset) for offset in max_offsets]
  File "mtrand.pyx", line 992, in mtrand.RandomState.randint
ValueError: Range cannot be empty (low >= high) unless no samples are taken
```
6. `NGINGIEN=SinZhug time dobi wavernn`，訓練WaveRNN模型。`4-ciidien-20190516-16k-MeuLid-checkpoints/hagfa_raw.wavernn`係模型，`5-ciidien-20190516-16k-MeuLid-model_outputs/hagfa_raw.wavernn`做得聽訓練時結果。程式愛走51點鐘。
```
4-ciidien-20190516-16k-MeuLid-checkpoints/
├── hagfa_lsa_smooth_attention.tacotron
│   └── ...
└── hagfa_raw.wavernn
    ├── latest_optim.pyt
    ├── latest_weights.pyt
    ├── log.txt
    ├── wave_step100K_optim.pyt
    ├── wave_step100K_weights.pyt
    └── ...

5-ciidien-20190516-16k-MeuLid-model_outputs/
├── hagfa_lsa_smooth_attention.tacotron
└── hagfa_raw.wavernn
    ├── 1000k_steps_1_gen_batched_target4000_overlap400.wav
    ├── 1000k_steps_1_target.wav
    ├── 1000k_steps_2_gen_batched_target4000_overlap400.wav
    ├── 1000k_steps_2_target.wav
    └── ...
```
7. `NGINGIEN=SinZhug time dobi habsang`，合成語句。
```
5-ciidien-20190516-16k-MeuLid-model_outputs/
├── hagfa_lsa_smooth_attention.tacotron
│   ├── __input_Kiung ha l_griffinlim_350k.wav
│   ├── __input_Kiung ha l_wavernn_batched_350k.wav
│   └── __input_Kiung ha l_wavernn_unbatched_350k.wav
└── hagfa_raw.wavernn
    └── ...

```

步用Intel i7-7700摎Nvidia 2080TI，走10分鐘以下，毋會寫時間。


### 定服務
```
docker-compose up --build
```

### 試合聲
Python3
```python3
from http.client import HTTPConnection
from urllib.parse import urlencode

參數 = urlencode({
    'toivun': 'Kiung ha loiˇ liau dong senˊ qi',
    'socoi': 'cii.wav',
})
headers = {
    "Content-type": "application/x-www-form-urlencoded",
    "Accept": "text/plain"
}
it_conn = HTTPConnection('localhost', port=5000)
it_conn.request("POST", '/', 參數, headers)
it_conn.getresponse().read()
```

### 結果
```
6-giedgo/
└── cii.wav
```

## 其他語言
### 設定
`camsu/hparams.py`裡度有：
```
CIIDIEN = {
    'MeuLid': ('四縣腔音讀', 's_sound'),
    'SinZhug': ('海陸腔音讀', 's_sound2'),
    'DungShe': ('大埔腔音讀', 's_sound3'),
    'SinZhu': ('饒平腔音讀', 's_sound4'),
    'Lun': ('詔安腔音讀', 's_sound5'),
    'LiugDui': ('南四縣腔音讀', 's_sound6'),
}
```

### 指令
頭前2步毋使改，共樣个。假使愛新竹話，第3步開始加`NGINGIEN`：
```
NGINGIEN=SinZhug time dobi preprocess-tacotron
NGINGIEN=SinZhug time dobi ...
```
