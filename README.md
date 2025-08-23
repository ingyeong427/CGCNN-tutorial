# 🖥️ CGCNN hands-on tutorial
CGCNN은 Jeffrey C. Grossman 교수님과 Tian Xie 박사님이 개발한 소재 물성 예측용 그래프 신경망 모델로, 이론적 배경은 다음 논문에 자세히 설명되어 있다.

- T. Xie and J. C. Grossman, Crystal graph convolutional neural networks for an accurate and interpretable prediction of material properties, Physical Review Letters 120,145301 (2018).
  
  <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301>


이 논문에서 다루고 있는 모델의 실습 코드는 <https://github.com/txie-93/cgcnn?tab=readme-ov-file>에서 다운받을 수 있다.

하지만 위 github의 데이터셋은 크기가 매우 작아 실습을 진행하는 데 부족함이 있기에, KIST 김동훈 박사님이 기존 github 데이터셋에 새로운 dataset과 보조기능 코드를 추가한 google colab 자료를 제공하고 있다. 

<https://drive.google.com/drive/folders/1HbxgZYCAJWynwFCwgWxfeg4-SrlWs0Gm>

이 튜토리얼은 상단의 dataset을 가지고 진행한 CGCNN 실습을 step-by-step으로 나타낸 것이다.
제공하는 데이터를 전부 다운로드 후, 압축 해제하면 된다. 




## 📌 프로그램 설치 및 환경설정
**CGCNN 실습을 위해서는 우리가 만들어 놓은 가상환경 안에 PyTorch, scikit-learn, pymatgen이라는 세 패키지를 설치해줘야 한다.
우선 conda 자체를 최신 버전으로 업데이트 하고, 가상환경을 만들어준 뒤 그 안에 패키지를 설치할 것이다.**


### **1. conda 업데이트**

conda가 최신버전이 아니라면, 패키지 설치 시 에러가 뜰 수도 있다.
anaconda prompt 프로그램을 실행시켜준 후, 다음처럼 입력하여 conda를 최신 버전으로 업데이트 시켜준다.
~~~
conda update -n base -c defaults conda
~~~
만일 업데이트를 했음에도 자꾸 
~~~
WARNING: A newer version of conda exists.
current version: 25.5.1
latest version: 25.7.0
~~~
처럼 뜬다면, 
~~~
conda update conda --all
~~~
을 입력하여 해결할 수 있다.


제대로 업데이트 되었는지는
~~~
conda --version
~~~
을 입력했을 때 최신 버전으로 뜨는지 확인하면 된다.


### **2. 가상환경 생성 후 그 안에 패키지 설치**

conda 전체가 아니라 실습을 진행할 환경에만 PyTorch, scikit-learn, pymatgen를 설치하기 위해 다음과 같이 새로운 가상 환경을 만들어준다. 

환경 이름은 원하는 대로 지어주면 된다. 이 튜토리얼에서는 'cgcnn'이라는 이름을 붙인 가상환경을 만들 것이다.
~~~
conda create -n cgcnn
~~~
기본환경인 (base) 대신 새로 생성된 (cgcnn)이라는 가상 환경을 활성화 시켜준 후,
~~~
conda activate cgcnn
~~~

다음과 같이 세 패키지를 설치해준다.
이때, pip은 파이썬으로 작성된 패키지 라이브러리를 설치하고 관리할 때 사용하는 명령어이다.
~~~
# PyTorch 설치
pip install torch

# scikit-learn 설치
pip install scikit-learn

# Pymatgen 설치
pip install pymatgen
~~~
`pip list`를 입력했을 때, 리스트에 세 패키지가 포함되어 있다면 성공적으로 설치된 이다.

이제 구글 드라이브에서 다운받았던 파일이 있는 경로로 이동해줘야 한다.
컴퓨터 상에서 구글 드라이브 폴더를 우클릭 하고 속성에 들어가면, 다음과 같이 파일 경로가 나온다.

~~~
C:\Users\ingyeong\Desktop\Summer
~~~

conda 내에서 이 파일에 접근하기 위해 prompt에 다음과 같이 경로를 입력해준다.
~~~
cd Desktop/Summer/cgcnn-master
~~~

다운받았던 파일이 있는 경로에 들어간 후, 그 폴더 안에 있던 코드인 main.py를 시험삼아 실행시켜 봤을 때 
~~~
python main.py -h
~~~
다음과 같이 option 목록이 쭉 뜬다면 실습을 위한 환경설정이 끝난 것이다.
~~~
usage: main.py [-h] [--task {regression, classification}]
.
.
~~~

## 📌 코드 Framework

파일은 서로 다음과 같은 관계를 가지며 작동한다. 각 파일에 대한 상세한 설명은 하단에 기술하였다. 

<img width="1578" height="869" alt="image" src="https://github.com/user-attachments/assets/bd64c46a-0b35-402d-943c-1982a24f756d" />


1) 하이퍼파라미터 설정 및 input 파일 구성하기
2) `main.py` 실행시키면 `id_prop.csv`를 읽어 첫 번째 열인 id의 목록을 얻게 됨.
3) `data.py`를 호출하여 `main.py`에서 읽어낸 id 목록을 `data.py`로 넘김. `data.py`는 받은 id 목록에 해당하는 `id.cif` 파일을 찾음.
4) `id.cif`으로부터 얻은 결정구조는 `atom_init.json` 파일을 바탕으로 벡터화된 그래프 형태로 나타내어짐.
5) `data.py`에서 벡터화된 그래프는 다시 `main.py`로 반환됨. 이후 `id_prop.csv`에 따라 벡터화된 결정구조 그래프와 물성이 매칭됨.
6) `main.py`는 하이퍼파라미터와 벡터화된 그래프 데이터를 `model.py`로 전송.
7) `model.py`는 모델의 CNN 구조를 구축한 후에 다시 `main.py`로 반환.
8) 정해진 epoch 횟수만큼 `main.py`을 통해 훈련 후, 훈련 결과 데이터 생성됨.
9) 훈련된 모델을 가지고 `predict.py` 진행 시, 예측 결과 데이터 생성됨.

------------------------

### 🔷 input 파일

Training과 Predicting을 위해 CGCNN 모델에 데이터를 입력하려면, 우선 필요한 데이터들을 하나의 폴더로 묶어야 한다. (customized dataset)

실습 파일 중에서는 sample-classification과 sample-regression이 customized dataset에 해당한다.

모델에 input으로 들어가는 customized dataset에는 다음 파일들이 포함되어야 한다.


- `id_prop.csv` : id와 property를 묶은 csv 파일로 1열에는 id, 2열에는 property가 적혀있다.
  
    - `id` : 각 결정구조마다 번호를 부여해준 것. (ex. cubic 구조의 SiO2 id는 8352)
  
    - `property` : 물성값(ex. bandgap, formation energy)을 의미.

       (Training 시)  재료의 실제 물성값을 입력해줘야 한다.
      
       (Predicting 시) 모델의 예측 성능을 평가하는 경우에는 실제 물성값을 입력해주어야 한다.

      예측만 하는 경우에는 실제 물성값이 필요 없으나, 2열을 비워둘 시 코드가 파일을 제대로 읽지 못하므로 아무 숫자(dummy)를 넣어서 형식을 맞춰주어야 한다.     
  
- `id.cif` : 결정구조에 대한 정보를 담고 있는 파일로, 결정의 물리적 특성, 좌표, 격자 등에 관한 정보를 알려준다.

   MP에서 제공하는 결정구조 파일은 'mp-id.cif' 형태로 제공된다. (ex. mp-13, mp-241)

   단지 우리가 input으로 활용하는 데이터인 'id_prop.csv'의 첫 열과 cif의 파일명을 일치시키기 위해 'id.cif'로 바꿔서 저장하는 것이다.
  
  
- `atom_init.json` : 원소를 숫자로 표현하기 위한 초기 벡터 데이터로, 주기율표를 기준으로 각 원소에 대한 특성이 one-hot encoding 된 형태로 정리되어 있다.

  쉽게 말해 Si는 벡터로 [숫자, , , ..], Na는 벡터로 [숫자, , , ..]와 같이 변환하라고 알려주는 참고용 문서이다.
  

### 🔷 모델 동작 파일 (.py 파일)

`.py` 파일은 마치 레시피라고 생각하면 된다. 우리가 레시피를 보고 요리하듯이, `.py` 파일의 코드를 실행시킴으로써 모델을 작동시키는 것이다.

- `main.py` : 결정구조(id)를 input으로 받아 물성(property)을 output으로 내놓는 CGCNN의 핵심 동작 파일로, `data.py`와 `model.py`를 연결해 train/predict를 수행한다.
  
- `data.py` : id를 input으로 받아 벡터화된 그래프를 output으로 내놓는다.
  
  입력받은 id에 해당하는 결정구조(.cif)를 받아오는 지점과, 결정구조를 보고 벡터화시키는 지점(atom_init.json)으로 구성되어 있다.
   
- `model.py` : pyTorch를 이용해 graph convolutional network 구조를 정의해준다.
- `predict.py` : 완성된 모델을 이용해 물성을 예측한다.
- `draw_graph.py` : 학습/예측 결과를 그래프로 나타내준다.

### 🔷 output 파일
- Training 결과 파일
  - `checkpoint.pth.tar` : 학습 중간 저장용 파일로, 마지막 epoch 모델 저장.
  - `model_best.pth.tar` : validation accuracy가 가장 높았던 모델 저장.
  - `epoch_loss.csv` : 각 epoch 마다의 훈련 loss값 기록.
  - `train_result.csv`, `validation_result.csv`, `test_result.csv` : train/validation/test set에서의 각 샘플별 예측 결과 기록.
  - `draw_graph.py` 실행 시 `epoch_loss.png`, `target_pred_test/train/validation.png` 파일 생성됨.
    
- Prediction 결과 파일
  - `test_result.csv` : test set으로 예측한 결과를 기록한 파일로 각 결정의 ID, 목표값(id_prop.csv 파일에서 넣어준 prop 값), 예측값(CGCNN이 예측한 값) 저장.

### 🔷 각 폴더 설명
- `data` : MP에서 가져온 train & predict를 위한 데이터 포함.
  - `sample-classification`, `sample-regression` : training을 위한 sample customized dataset.

  - `data_classification/regression_` : 각 물성값을 훈련하기 위한 데이터셋.
    
- `node_vector_generation` : node feature vector 수정을 위한 파일 포함.
   
- `pre-trained` : 논문에서 다루고 있는 pre-trained 모델에 대한 data 포함.
  
- `result` : `data` 폴더에 있는 데이터셋으로 훈련/예측한 결과값.

## 📌 각종 parameter 조절법

#### 🔷 hyperparameter

hyperparameter 조절 방식은 다음과 같다.
~~~
python main.py [데이터셋 폴더 경로] [hyperparameter 수정 옵션]
~~~
예시 코드는 다음과 같다.
~~~
python main.py data/sample-classification --epochs 1200 --n-conv 5 --lr 0.03 
~~~

#### 🔷 node feature vector

node feature vector에 대한 정보는 `atom_init.json` 파일에 저장되어 있다.
  
만일 node feature vector를 조절하고 싶다면, `encoding_feature_num.py`를 수정하면 된다.  

7번째 줄부터 나와있는 feature list 중, 실제로 node feature vector에 활용하고자 하는 feature들을 선택해 12번째 줄의 feature set에 입력하면 된다.

`encoding_feature_num.py`가 수정되면 자동으로 `atom_init.json` 파일도 덮어쓰기 모드로 수정되어 node vector가 조절된다.

#### 🔷 edge feature vector

`data.py` 파일을 수정하면 edge vector를 조절할 수 있다. 

`data.py`의 275~289번째 줄에서는 edge vector와 관련된 hyperparameter들을 설명하고 있다.

~~~
- root_dir (str) : 어떤 데이터셋 폴더를 사용할 것인가
- max_num_nbr (int) : 결정 그래프를 형성할 때 몇 개의 이웃 원자까지만 연결할 것인가
- radius (float) : 얼마의 반지름 이내에 있는 원자만 이웃 원자로 정의할 것인가
- dmin (float) : 두 원자 사이의 거리가 최소 얼마 이상이라고 가정할 것인가 (가우시안 벡터화를 위한 최소 길이)
- step (float) : edge vector를 몇 칸 간격으로 쪼갤 것인가 (가우시안 basis의 간격) 
- random_seed (int) : 데이터를 섞을 때 고정하는 난수의 개수
~~~

이 parameter들을 수정하여 edge vector를 조절하려면 300번째 줄의 값들(radius,dmin, ..)을 조절하면 된다. (Gaussian distancing 형태의 edge vector)

~~~
def __init__(self, root_dir, max_num_nbr=12, radius=8, dmin=0, step=0.2,
             random_seed=123):
~~~

## 📌 Training by txie-93 github dataset

우선 sample-regression 폴더에 있는 적은 수의 데이터를 가지고 훈련해볼 것이다. 

단지 코드 동작을 확인하기 위해 구성해놓은 dataset이기 때문에, sample-regression과 sample-classification 폴더에 있는 id_prop.csv 파일의 prop 값들은 실제 물성값이 아닌 dummy 값이다.

main.py는 'cgcnn-master' 폴더에 들어있기 때문에 다음과 같이 이 폴더의 경로에서 시작해야 한다.

~~~
(cgcnn) C:\Users\ingyeong\Desktop\Summer\cgcnn-master>
~~~

main.py를 실행시킬 때는 train, validation, test의 ratio 혹은 size와, 어느 경로에 있는 데이터를 사용할지를 지정해주면 된다. 

이때 ratio와 size는 혼용하면 안된다.

~~~
python main.py --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 data/sample-regression
~~~
or
~~~
python main.py --train-size 6 --val-size 2 --test-size 2 data/sample-regression
~~~

훈련이 끝나면 cgcnn-master 폴더에 `checkpoint.pth.tar`, `model_best.pth.tar`, 그리고 각종 `.csv`파일들이 저장된다. 

이 결과들에 대해서 그래프를 그리고 싶다면, 다음 코드를 실행시키면 된다.
~~~
python draw_graph.py
~~~
생성된 그래프들은 cgcnn-master 폴더에 png 파일로 저장된다.

## 📌 Predicting by txie-93 github dataset

Predicting은 `predict.py` 코드를 이용하여 진행된다. 이 튜토리얼에서는 논문에 나오는 미리 훈련된 모델인 `pre-trained` dataset을 활용할 것이다. 예측하고자 하는 물성에 따라 해당하는 폴더를 사용하면 된다.

예를 들어 `sample-regression` 폴더에 있는 결정의 formation energy를 예측하고 싶다면, 다음과 같이 코드를 작성하면 된다.

~~~
python predict.py pre-trained/formation-energy-per-atom.pth.tar. data/sample-regression
~~~

또다른 예시로 'sample-classification' 폴더에 있는 결정들에 대해 반도체면 (0), 도체면 (1)로 예측하고 싶다면 다음과 같이 코드를 작성하면 된다.

~~~
python predict.py pre-trained/semi-metal-classification.pth.tar. data/sample-classification
~~~

예측에 대한 결과 데이터들은 `test_results.csv` 파일로 저장된다.

## 📌 Training by google drive dataset

txie-93 github에서 제공하는 샘플 데이터셋은 크기가 매우 작기때문에, KIST의 김동훈 박사님이 customized dataset을 구글 드라이브에 제공하고 있다. 

`data` 폴더에 들어있는 'data_' 이름의 하위 폴더들이 전부 customized dataset에 해당한다. 

폴더명에는 각각 학습하고자 하는 물성과 cif 파일의 개수가 쓰여있고, bandgap 학습용 데이터셋은 metal과 non-metal의 cif 파일 개수가 1:1 비율로 구성되어 있다.
  
훈련은 이전에 진행한 것과 동일한 방식으로 진행하면 된다.

~~~
python main.py --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 data/data_regression_formE_1000
~~~

그래프를 그리는 코드 또한 동일하다.
~~~
python draw_graph.py
~~~
