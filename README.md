# 🖥️ CGCNN-tutorial
이 튜토리얼은 <https://colab.research.google.com/drive/19habZtpW5luFECqRy7WrjDa1keAj2Xyg#scrollTo=gbppBq-AipwL>의 dataset을 가지고 진행한 CGCNN 실습을 step-by-step으로 나타낸 것이다.

이 파일은 논문 저자가 제공하고 있는 dataset인 <https://github.com/txie-93/cgcnn?tab=readme-ov-file>을 포함하고 있다.  

위 github의 데이터셋은 크기가 매우 작아 실습을 진행하는 데 부족함이 있기에, KIST 김동훈 박사가 새로운 dataset과 보조기능 코드를 추가한 colab 자료를 이용하는 것이다.  

하단의 구글 드라이브에서 제공하는 데이터를 전부 다운로드 후, 압축 해제하면 된다. 
<https://drive.google.com/drive/folders/1HbxgZYCAJWynwFCwgWxfeg4-SrlWs0Gm>

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
생성된 'cgcnn'이라는 이름의 가상 환경 안안에 세 패키지를 설치해준다.
이때, pip은 파이썬으로 작성된 패키지 라이브러리를 설치하고 관리할 때 사용하는 명령어이다.
~~~
# PyTorch 설치
pip install torch

# scikit-learn 설치
pip install scikit-learn

# Pymatgen 설치
pip install pymatgen
~~~
pip list를 입력했을 때, 리스트에 세 패키지가 포함되어 있다면 설치 성공이다.

이제 구글 드라이브에서 다운받았던 파일이 있는 경로로 이동해줘야 한다.
컴퓨터 상에서 구글 드라이브 폴더를 우클릭 하고 속성에 들어가면, 다음과 같이 파일 경로가 나온다.

<img width="349" height="42" alt="image" src="https://github.com/user-attachments/assets/a2af97f6-29a4-4e92-980a-2edae9e5edf9" />

경로 확인 후, cd 명령어를 이용해 파일 경로까지 거슬러 올라가준다.
~~~
cd Desktop/Summer/google_drive_CGCNN
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

### 각 파일 설명

#### input 파일
- `id.cif` : 결정구조에 대한 정보를 담고 있다. 쉽게 말해 결정에 원자가 어떠한 방식으로 배치되어 있는지 좌표/격자에 대한 정보를 알려준다.
  
- `atom_init.json` : 원소를 숫자로 표현하기 위한 초기 벡터 데이터로, 주기율표를 기준으로 각 원소에 대한 특성이 one-hot encoding 된 형태로 정리되어 있다.

  쉽게 말해 Si는 벡터로 [숫자, , , ..], O는 벡터로 [숫자, , , ..]와 같이 변환하라고 알려주는 참고용 문서이다.
  
- `id_prop.csv` : id와 property를 묶은 csv 파일로 1열에는 id, 2열에는 property가 적혀있다.
  
    `id`란 데이터셋 안에서 각 결정 구조를 구분하는 식별자로, 말 그대로 각 결정구조에 번호를 부여해준 것이라고 생각하면 된다.
  
    `prop`란 예측하려는 물성값(ex. bandgap, formation energy)을 의미한다.

  학습할 때는 2열의 물성값이 정답으로 쓰이지만, 예측할 때는 정답 값이 필요 없다.

  하지만 2열을 비워둘 시 코드가 파일을 제대로 읽지 못하므로, 아무 숫자라도 넣어서 형식을 맞춰줘야 한다.

#### .py 파일

- `main.py` : input과 output을 가지고 학습한다. 이때 input은 'Materials Project(MP)'의 id이고, output은 bulk property이다.
- `data.py` : main.py로부터 받은 id를 bulk structure(회색)로 넘겨 구조를 얻어냄. 이후 다시 main.py에게 벡터화 된 그래프 형태로 넘겨줌.

  이걸 조절하면 edge vector와 관련된 hyperparameter들을 조절할 수 있음.???
  
- `model.py` : graph convolution에 필요한 class들이 들어가있음.
- `predict.py` :
- `draw_graph.py` : 결과를 그래프로 나타내주는 코드이다.
- `__init__.py` : 왜 있는지 모르겠음;;;;
- `.pth` :
- `mp-ids.csv` :
- `mp-id.cif` : MP에서 제공하는 결정 구조 파일


### 각 폴더 설명
- `data` : MP에서 가져온 train & predict를 위한 데이터가 들어가 있다.
  - `sample-classification`, `sample-regression` :
  
    Training과 Predicting 전에, CGCNN에게 입력할 데이터들을 하나의 폴더로 모아놓아야 한다. 

    모아놓은 이 input 폴더에는 `id_prop.csv`, `atom_init.json`, `ID.cif` 파일이 들어가 있어야 한다.

    우리가 다운받은 샘플 코드 중에서는 data 폴더 안에 있는 `sample-classification` 폴더와 `sample-regression` 폴더가 이 input 폴더에 해당한다.
- `node_vector_generation` : 이 폴더 내에서 작업 시, node feature vector를 수정할 수 있다.
  
   node feature vector에 대한 정보는 `atom_init.json` 파일에 저장되어 있다.
  
   만일 node feature vector를 수정하고 싶다면, `encoding_feature_num.py` 코드 수정 시 `atom_init` 파일도 덮어쓰기 모드로 수정된다.
   
- `pre-trained` : 논문에서 보고되었던 학습된 모델에 대한 data가 들어가 있다.
- `checkpoint.pth` : 반복 학습하며 가장 좋았던 모델을 백업해 놓음.
- `model_best.pth` : 학습 중 가장 좋은 모델을 저장해 놓음.



-------------------------
Materials Project 사이트에서는 보통 각 결정구조를 고유한 materials_id로 관리한다. (mp-13, mp-241 이런식으로..)
ID.cif 파일은 보통 그 ID에 해당하는 결정구조의 좌표를 저장하고 있다.
id_prop.csv 파일은 그 ID에 대응하는 물성값을 저장하고 있다.





-----------------------------
main.py만 돌리면서 어떤 폴더에 있는 데이터를 쓰라는 것만 지정해주면 됨. (data/sample-regression)

~~~
python main.py --train-size 6 --val-size 2 --test-size 2 data/sample-regression
~~~
python에게 main.py 파일을 돌려라! ratio는 다음과 같이! 우리가 train 하고자 하는 데이터의 폴더 location은 다음과 같다!
