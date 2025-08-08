# CGCNN-tutorial
이 튜토리얼은 
<https://colab.research.google.com/drive/19habZtpW5luFECqRy7WrjDa1keAj2Xyg#scrollTo=gbppBq-AipwL>의 data를 가지고 진행한 CGCNN 실습을 step-by-step으로 나타낸 것이다.

하단의 구글 드라이브에서 제공하는 데이터를 전부 다운로드 후, 압축 해제하면 된다.
<https://drive.google.com/drive/folders/1HbxgZYCAJWynwFCwgWxfeg4-SrlWs0Gm>


## :pushpin: 프로그램 설치 및 환경 설정
★CGCNN을 활용하기 위해서는 PyTorch, scikit-learn, pymatgen이라는 세 패키지지를 설치해줘야 한다.★


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

conda 전체가 아니라 필요한 환경에만 PyTorch, scikit-learn, pymatgen를 설치하기 위해 다음과 같이 'cgcnn'이라는 이름을 붙인 가상 환경을 구성해준다.
~~~
conda create -n cgcnn
~~~
생성된 'cgcnn'이라는 이름의 가상 환경에 세 패키지를 설치해준다.
이때, pip 명령어는 파이썬으로 작성된 패키지 라이브러리를 설치하고 관리할 때 사용한다.
~~~
# PyTorch 설치
pip install torch

# scikit-learn 설치
pip install scikit-learn

# Pymatgen 설치
pip install pymatgen
~~~
pip list를 입력했을 때, 리스트에 세 패키지가 포함되어 있다면 설치 성공이다.

구글 드라이브에서 다운받았던 파일이 있는 경로로 이동해줘야 한다.
컴퓨터 상에서 구글 드라이브 폴더를 우클릭 하고 속성에 들어가면, 다음과 같이 파일 경로가 나온다.

<img width="349" height="42" alt="image" src="https://github.com/user-attachments/assets/a2af97f6-29a4-4e92-980a-2edae9e5edf9" />

경로 확인 후, cd 명령어를 이용해 파일 경로까지 거슬러 올라가준다.
~~~
cd Desktop
cd summer
cd google_drive_CGCNN
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

## 논문 원작자 코드 실습습
