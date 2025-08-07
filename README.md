# CGCNN-tutorial
이 튜토리얼은 
<https://colab.research.google.com/drive/19habZtpW5luFECqRy7WrjDa1keAj2Xyg#scrollTo=gbppBq-AipwL>의 data를 가지고 진행한 CGCNN 실습을 step-by-step으로 나타낸 것이다.

하단의 구글 드라이브에서 제공하는 데이터를 전부 다운로드 후, 압축 해제하면 된다.
<https://drive.google.com/drive/folders/1HbxgZYCAJWynwFCwgWxfeg4-SrlWs0Gm>

## 프로그램 설치 및 환경 설정
1. CGCNN을 활용하기 위해서는 PyTorch, scikit-learn, pymatgen이라는 세 라이브러를 설치해줘야 한다.

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
을 통해 해결해줄 수 있다.


제대로 업데이트 되었는지는
~~~
conda --version
~~~
을 통해 확인하면 된다.
-----------------
~~~
#PyTorch 설치
!pip install torch

# scikit-learn 설치
!pip install scikit-learn

# Pymatgen 설치
!pip install pymatgen
~~~
