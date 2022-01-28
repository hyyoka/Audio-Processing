# Audio-Processing

## 1. Audio Augment

### Conventional apporach
1. change_pitch
2. value_aug
3. add_noise
4. shift
5. hpss
6. change_pitch_and_speed
7. resample
8. time_stretching

### SpecAugment
SpecAugment는 위에서 제시한 전통적인 audio augmentation처럼 audio waveform을 변형하는 방식이 아니라, 데이터를 audio가 아닌 image처럼  여기는 대신, 음성의 특성을 고려하면서 augment를 하는 방식이다. 사용하는 방식은 3가지이다.
1. Time wraping
    - Computer Vision에서 사용되는 Image Warping을 응용한 방법으로, 축의 중심을 이동하여 데이터 수를 늘린다. 
2. Frequency Masking
    - 주파수와 시간 축으로 이루어진 Spectrogram의 주파수 축을 따라 일정 영역을 0으로 마스킹
3. Time Masking
    - 시간 축에 대해서 일정 영역을 0으로 마스킹

### Specmix

1. specmix


[Comparing data augmentation for audio classification](https://iopscience.iop.org/article/10.1088/1742-6596/1453/1/012085/pdf) 


논문에서 각 augment 기법들에 따른 성능 향상 정도를 비교해놓은 표가 있어 첨부한다. 
![스크린샷_2021-12-10_오후_2.43.28](https://i.imgur.com/azlT1py.png)

- SpecAugment: time warping, frequency masking, time masking
- mixup: weighted average of two samples
- SpecAugment: SpecAugment + Mixup

![all_augmentations](https://i.imgur.com/oZ4VPmn.png)



## 2. Speaker Diarlization
1. speaker diarlization -> combine uniform speaker's wv
2. speaker diarlization -> mask other speaker's utterance

Speaker Diarization이란 어떤 화자가 언제 말했는지를 파악하는 Task이다. 

현재 Speaker Diarization task의 sota 모델은 당연히 self-attention based의 딥러닝 모델이다. 하지만, 이는 크기가 매우 크기도 하고, 공개되어있는 Pretrained model도 희소하며, 한국어 모델이 없다(한국어 모델이 필요한지는 의문이다). 그렇기에 전체적인 흐름을 보면서 어떤 접근법이 사용되었는지를 우선적으로 파악하고, 쓸만한 모델들을 확인한다.

### Conventional apporach


![스크린샷_2021-12-23_오후_5.21.49](https://i.imgur.com/NpmkukK.png)

전통적인 방식은 상단과 같다. 
- Front-end processing
    - audio 파일에 있는 noise를 억압하는 방식으로 학습된 Speech Enhancement/Denoising module
    - STFT값과 impulse값을 곱해 noise를 더하면 original signal x를 근사하는 x'을 만들 수 있는데, 이 x'과 log power spectrum s 사이의 loss를 최소화하는 방식으로 학습된다.  

- Speech activity Detection (SAD)
    - speeech와 non-speech를 구분하는 module. 
    - non-speech의 예시로는 background noise가 있다. 
    - 크게 두 가지 부분으로 나누어진다. 
        1. Feature Extract: zero-crossing-rate, pitch, mfcc, etc
        2. Classifier: 베이스 기반 모델부터 딥러닝 모델까지.
    - 전체적인 모델의 성능에 큰 영향을 끼치는 부분(cascaded 모델은 앞선 오류가 propagate)

- Segmentation
    - speaker uniform segment로 audio를 자르는 부분
    - 두 가지 접근법 존재
        1. Speaker change point detection
             - 해당 방법론은 decision boundary를 학습하는 것이라 할 수 잇음
             - 하지만 segment들의 길이가 너무 다양해서 견고한 학습이 어려움
        2. Uniform segmentation
            - 정해진 길이로 오디오를 자름, 이때 발화자가 겹치지 않을 정도의 segment 길이 필요
            - trade-off존재: speaker representation을 잘 할 수 있지만, 겹치게 하지 않는 길이를 찾아야함. 
- Speaker Embedding
    - Joint factor analysis (JFA):  

        - ![스크린샷_2021-12-24_오전_11.44.43](https://i.imgur.com/hI6pXGD.png)
    - i-vector: 낮은 차원의 w를 이용해 conversation side를 표현. 각 factor는 행렬 T의 eigen-dimension을 제어하며, 이를 i-vector라고 한다 
        - i-vector = speaker representation vector
        - ![스크린샷_2021-12-24_오전_11.44.43](https://i.imgur.com/YGZrYEg.png)
        - i-vector를 발전시키기 위해 적용할 수 잇는 것이 PLDA
    - x-vector: embedding vector representations based on the bottleneck layer output of a deep neural network (DNN) trained for speaker recognition
        - 일반적으로 PLDA를 베이스로 하는 x-vector를 사용함.
    - 딥러닝을 단독으로 이용하는 방법도 등장했으나, 성능이 i-vector와 이를 발전시킨 x-vector를 사용하는 것보다 좋지 않았다. 둘을 함께 사용할 때 성능이 괜찮았음. 
- Clustering
    - 위의 speaker representation를 고려하여 speech segment들을 clustering한다. 모인 segment는 하나의 화자가 말한 발화들이 된다. 
    - Agglomerative Hierarchical Clustering과 Spectral Clustering이 주로 사용되며, 그외 k-means와 같은 클러스터링 기법들은 좋은 성능을 내지 못했다고 한다. 
    - 최근에는 GNN(graph)를 이용한 클러스터링 방법론도 사용된다. 
- Post Processing
    - resegmentation: 모델=VB-HMM
        - segmentation과 clustering을 함께 최적화하는 모델 
  
### Deep learning Approach

- deep learning for Clustering

앞서 언급했듯, GNN(graph neural network)을 이용해 spectral clustering의 결과물을 보정하는 방법도 존재하며, belonging factor를 clustering 기법에 추가한 deep embedded clustering(DEC)도 사용한다. DEC의 성능을 보다 끌어올리기 위해 몇 가지 error term을 추가한다. 

![스크린샷_2021-12-24_오후_1.10.11](https://i.imgur.com/XvyUcTg.png)

최종 loss는 clustering error + reconstruction error + uniform spearker airtime distribution error + distance from centroid bottleneck regularization으로 이루어진다. 


- Deep learning for Distance 
이러한 방법 말고, approach for learning the relationship between the speaker cluster centroids (or speaker profiles) and
the embedding도 존재. 

예를 들어 RNN을 이용하는 경우, 각 segment들은 어떤 화자인지에 대한 classification task로 전환된다. 이때 거리를 줄인다는 의미는 모든 segment와 화자 label 사이의 거리를 줄인다는 의미에서, relationship between embeddings and profile이라고 표현할 수 있다. 


- Fully E2E Neural Diarization

위에서는 전통적인 모델들의 모듈 성능을 향상시키기 위해 노력했다면, 현재의 트렌드는 EEND, 즉 음성신호를 넣어주면 처음부터 끝까지 하나의 모델이 diarization을 수행하는 것이다. 

input: T-length sequence of acoustic features (ex. log Mel-filterbank)
output: 한 화자에 대해 발화하면 1 조용하면 0으로 태깅된 배열 반환 
이때 중요한 점은 **여러 명의 화자가 동시 발화하는 경우까지 고려한다는 것**. 
예를 들어 두 화자는 다음과 같이 동시 발화를 할 수 있다. 
```
[0,0,0,0,0,0,1,1,1,1,0,0,0,1,1,1,1,1] 
[0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1] 
```

이렇게 한 음성의 overlapping problem을 해결하는 task를 speech separation이라고 한다. 위와 같은 방식으로 태깅을 함으로써 이 문제까지 함께 E2E로 해결할 수 있다. 

하지만 LSTM이든 self-attention-based이든, 대응 가능한 화자의 수가 한정되어있다는 단점이 있었는데, 이를 극복하기 위한 모델이 encoder decoder-based attractor (EDA)이다. LSTM을 기반으로 만들어진 이 모델은, output에 attractor를 생성하도록 한다. 이 attractor는 원래의 output과 곱해져서 화자의 speech activity를 계산한다. 


### 사용 Framework

| Link | Language | Description |
| ---- | -------- | ----------- |
| [pyBK](https://github.com/josepatino/pyBK) ![GitHub stars](https://img.shields.io/github/stars/josepatino/pyBK?style=social) | Python | Speaker diarization using binary key speaker modelling. Computationally light solution that does not require external training data. |
