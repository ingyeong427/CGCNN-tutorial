# 🖥️ CGCNN-tutorial
CGCNN은 Jeffrey C. Grossman 교수님과 Tian Xie 박사님이 개발한 소재 물성 예측용 그래프 신경망 네트워크 모델로, 다음 논문에 자세히 설명되어 있다.

<https://doi.org/10.1103/PhysRevLett.120.145301>

이 논문에서 다루고 있는 모델의 실습 코드는 <https://github.com/txie-93/cgcnn?tab=readme-ov-file>에서 다운받을 수 있다.

하지만 위 github의 데이터셋은 크기가 매우 작아 실습을 진행하는 데 부족함이 있기에, KIST 김동훈 박사님이 기존 github 데이터셋에 새로운 dataset과 보조기능 코드를 추가한 google colab 자료를 제공하고 있다. 

<https://drive.google.com/drive/folders/1HbxgZYCAJWynwFCwgWxfeg4-SrlWs0Gm>

이 튜토리얼은 상단의 dataset을 가지고 진행한 CGCNN 실습을 step-by-step으로 나타낸 것이다.
구글 드라이브에서 제공하는 데이터를 전부 다운로드 후, 압축 해제하면 된다. 




## 📌 프로그램 설치 및 환경설정
**CGCNN 실습을 위해서는 우리가 만들어 놓은 가상환경 안에 PyTorch, scikit-learn, pymatgen이라는 세 패키지를 설치해줘야 한다.
우선 conda 자체를 최신 버전으로 업데이트 하고, 가상환경을 만들어준 뒤 그 안에 패키지를 설치할 것이다.**


### **1. conda 준비**

anaconda prompt 프로그램을 실행시켜준 후, 맨 앞이 (base)로 시작하는 것을 확인하고 다음처럼 입력하여 아나콘다를 최신 버전으로 업그레이드 시켜준다.
~~~
conda update -n base -c defaults conda
~~~
만일 업데이트를 했음에도 자꾸 
~~~
WARNING: A newer version of conda exists.
current version: 25.5.1
latest version: 25.7.0
~~~
와 같이 뜬다면, 
~~~
conda update conda --all
~~~
을 입력하여 해결할 수 있다.


제대로 업데이트 되었는지는
~~~
conda --version
~~~
을 통해 확인하면 된다.


### **2. 가상환경 만들고 그 안에 패키지 설치**

conda 전체가 아니라 필요한 환경에만 PyTorch, scikit-learn, pymatgen를 설치하기 위해 다음과 같이 'cgcnn'이라는 이름을 붙인 가상 환경을 만들어준다.
~~~
conda create -n cgcnn
~~~
생성된 'cgcnn'이라는 이름의 가상 환경 안에 들어가준 후,
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
`pip list`를 입력했을 때, 리스트에 세 패키지가 포함되어 있다면 설치 성공이다.

이제 구글 드라이브에서 다운받았던 파일이 있는 경로로 이동해줘야 한다.
컴퓨터 상에서 구글 드라이브 폴더를 우클릭 하고 속성에 들어가면, 다음과 같이 파일 경로가 나온다.

<img width="349" height="42" alt="image" src="https://github.com/user-attachments/assets/a2af97f6-29a4-4e92-980a-2edae9e5edf9" />

경로 확인 후, cd 명령어를 이용해 파일 경로까지 거슬러 올라가준다.
~~~
cd Desktop/Summer/cgcnn-master
~~~

다운받았던 파일이 있는 경로에 들어간 후, 그 폴더 안에 있던 코드인 main.py를 시험삼아 실행시켜 봤을 때 
~~~
python main.py -h
~~~
다음과 같이 뜬다면 모든 준비가 끝난 것이다.
~~~
usage: main.py [-h]
.
.
~~~

## 📌 코드 Framework

파일은 서로 다음과 같은 관계를 가지며 작동한다. 각 파일에 대한 상세한 설명은 하단에 기술하였다. 

  ![그림1](https://github.com/user-attachments/assets/6919bb6d-28dc-47a7-9d87-fb076392c278)



1) input 파일 및 하이퍼파라미터 설정
2) `main.py` 실행시키면 `id_prop.csv`를 읽어 첫 번째 열인 id의 리스트를 얻게 됨.
3) `data.py`를 호출하여 `main.py`에서 읽어낸 id 리스트가 `data.py`로 넘어감. `data.py`는 받은 id 리스트에 해당하는 `id.cif` 파일을 찾음.
4) `id.cif`에서 얻은 결정구조는 `atom_init.json` 파일을 바탕으로 벡터화된 그래프 형태로 나타내짐.
5) `data.py`에서 벡터화된 그래프는 다시 `main.py`로 반환됨. 이후 `id_prop.csv`에 따라 결정구조와 물성이 매칭됨.
6) `main.py`는 하이퍼파라미터와 벡터화된 그래프 데이터를 `model.py`로 전송.
7) `model.py`는 CGCNN 모델의 구조를 정의한 후에 다시 `main.py`로 반환.
8) `main.py`에서 정해진 epoch 횟수만큼 학습 진행 후, 결과 데이터 생성.

------------------------

### 🔷 input 파일

- `id_prop.csv` : id와 property를 묶은 csv 파일로 1열에는 id, 2열에는 property가 적혀있다.
  
    `id`란 각 결정 구조를 구분하는 식별자로, 말 그대로 각 결정구조에 번호를 부여해준 것이라고 생각하면 된다.
  
    `prop`란 예측하려는 물성값(ex. bandgap, formation energy)을 의미한다.

  학습할 때는 2열의 물성값이 정답으로 쓰이지만, 예측할 때는 정답 값이 필요 없다.

  하지만 2열을 비워둘 시 코드가 파일을 제대로 읽지 못하므로, 아무 숫자라도 넣어서 형식을 맞춰줘야 한다.

- `id.cif` : 결정구조에 대한 정보를 담고 있다. 쉽게 말해 결정에 원자가 어떠한 방식으로 배치되어 있는지 좌표/격자에 대한 정보를 담고있다.
  
  참고) Materials Project 에서는 각 결정구조를 고유한 숫자인 'id'로 관리한다. (ex. mp-13, mp-241)

   MP에서 제공하는 결정구조 파일은 mp-id.cif 형태로 제공된다.

   단지 우리가 input으로 활용하는 데이터인 id_prop.csv의 첫 열과 cif 파일명을 일치시키기 위해 id.cif로 저장하는 것이다.
  
- `atom_init.json` : 원소를 숫자로 표현하기 위한 초기 벡터 데이터로, 주기율표를 기준으로 각 원소에 대한 특성이 one-hot encoding 된 형태로 정리되어 있다.

  쉽게 말해 Si는 벡터로 [숫자, , , ..], O는 벡터로 [숫자, , , ..]와 같이 변환하라고 알려주는 참고용 문서이다.

### 🔷 모델 동작 파일

.py 파일은 쉽게 말해 레시피라고 생각하면 된다. 우리가 레시피를 보고 요리하듯이, .py 파일을 실행시킴으로써 모델이 작동할 수 있는 코드를 동작시키는 것이다.

- `main.py` : CGCNN의 핵심 원리가 구현되는 코드로, MP로부터 결정구조(id)를 input으로 받아 물성(property)을 output으로 내놓는다.

  참고) 기타 하이퍼파라미터도 `main.py -h`를 입력하면 조절할 수 있다.
  
- `data.py` : id를 input으로 받아 벡터화된 그래프를 output으로 내놓는다.
  
  입력받은 id에 해당하는 결정구조(.cif)를 받아오는 지점과, 결정구조를 보고 벡터화시키는 지점(atom_init.json) 으로 구성되어 있다.
  
  참고) data.py 코드 중 300번째 줄의 값들(radius,dmin, ..)을 조절하면 edge vector 조절이 가능하다. (Gaussian distancing 형태의 edge vector)

 ~~~
def __init__(self, root_dir, max_num_nbr=12, radius=8, dmin=0, step=0.2,
              random_seed=123):
~~~
   
- `model.py` : graph convolution에 필요한 정보들이 pytorch 패키지를 사용해 만들어져 있다.
- `predict.py` : 완성된 모델을 이용해 물성을 예측한다.
- `draw_graph.py` : 학습/예측 결과를 그래프로 나타내준다.

### 🔷 output 파일
- `checkpoint.pth.tar` : 마지막 epoch 모델 저장.
- `model_best.pth.tar` : 학습 중 가장 정확한 validation 결과를 낸 모델 저장.
- `test_result.csv` : test set에 있는 각 결정의 ID, target value, predicted value 저장.

### 🔷 각 폴더 설명
- `data` : MP에서 가져온 train & predict를 위한 데이터가 들어가 있다.
  - `sample-classification`, `sample-regression` :
  
    Training과 Predicting 전에, CGCNN에게 입력할 데이터들을 하나의 폴더로 모아놓아야 한다. 

    모아놓은 이 input 폴더에는 `id_prop.csv`, `atom_init.json`, `ID.cif` 파일이 들어가 있어야 한다.

    우리가 다운받은 샘플 코드 중에서는 data 폴더 안에 있는 `sample-classification` 폴더와 `sample-regression` 폴더가 이 input 폴더에 해당한다.
    
- `node_vector_generation` : 이 폴더 내에서 작업 시, node feature vector를 조절할 수 있다.
  
   node feature vector에 대한 정보는 `atom_init.json` 파일에 저장되어 있다.
  
   만일 node feature vector를 수정하고 싶다면, `encoding_feature_num.py` 의 feature set을 조절 시 `atom_init.json` 파일도 덮어쓰기 모드로 수정된다
   
- `pre-trained` : 논문에서 보고되었던 학습된 모델에 대한 data가 들어가 있다.

## 📌 샘플 데이터 훈련 (txie-93 github version)

우선 적은 데이터를 가지고 훈련해보기 위해, sample-regression이라는 폴더에 input 데이터를 모두 구성해놓았다.

main.py는 'cgcnn-master' 폴더에 들어있기 때문에 다음과 같이 이 폴더의 경로에서 시작해야 한다.

~~~
(cgcnn) C:\Users\ingyeong\Desktop\Summer\cgcnn-master>
~~~

main.py를 실행시킬 때는 다음과 같이 train : validation : test의 비율과, 어느 경로에 있는 데이터를 사용할지를 지정해주면 된다. 

~~~
python main.py --train-size 0.6 --val-size 0.2 --test-size 0.2 data/sample-regression
~~~
현재 위치는 cgcnn-master 폴더인데, 이 폴더는 cgcnn-master/data/sample-regression에 있기 때문에 data/sample-regression 라고 적어주는 것이다.

훈련 결과에 대해서 그래프를 그리고 싶다면, 다음 코드를 실행시키면 된다.
~~~
python draw_graph.py
~~~
생성된 결과와 그래프들은 cgcnn-master 폴더에 csv 파일과 png 파일로 저장되어 있을 것이다.


## 📌 customized dataset 훈련 (google colab version)

github에서 제공한 샘플 데이터셋은 크기가 매우 작기때문에, customized dataset 3가지를 제공한다. 데이터 수집 방식은 다음과 같다.

- Materials Project에서 조건을 is_stable=True (energy_above_hull = 0), 삼원계 이하, 그리고 포함원소의 원자번호를 Bi 이하 (noble gas 원소 제외) 조건으로 검색하면 22,962개의 구조-물성 (formation energy, bandgap) 데이터를 획득
- 여기서 랜덤하게 1,000개를 뽑아서 아래의 데이터셋을 구성한다.
- Band gap의 경우, metal 500개, non-metal 500개임.
  
  <img width="1908" height="685" alt="download" src="https://github.com/user-attachments/assets/78714a78-efc5-4732-8a2d-8936756709e2" />
  
**Dataset #1. regression (formation energy)**

data size : 1000

Path : /data/data_regression_formE_1000/

**Dataset #2. regression (band gap)**

data size : 1000

Path : /data/data_regression_bandgap_1000/

**Dataset #3. classification (band gap)**

data size : 1000

Path : /data/data_classification_metal_1000/

훈련은 github 코드와 동일한 방식으로 다음과 같이 진행하면 된다.

~~~
python main.py --train-size 0.6 --val-size 0.2 --test-size 0.2 data/data_regression_formE_1000
~~~

그래프를 그리는 코드 또한 동일하다.
~~~
python draw_graph.py
~~~

## 📌 훈련된 CGCNN 모델을 가지고 물성 예측

`pre-trained` 폴더에는 논문에 나오는 훈련된 CGCNN 모델이 들어있다. 이 모델을 `predict.py`로 실행시키면 물성을 예측할 수 있다.

예를 들어 `sample-regression` 폴더에 있는 결정의 formation energy를 예측하고 싶다면, 다음과 같이 코드를 작성하면 된다.

~~~
python predict.py pre-trained/formation-energy-per-atom.pth.tar. data/sample-regression
~~~

결과 데이터는 `test_results.csv` 파일로 저장된다.
