# RDD AI Image Detection

**추가 학습 없이 사전학습 확산 모델의 복원 오차로 AI 생성 이미지를 탐지하는 개인 연구 프로젝트.**

jaewon1007의 Kaggle `LG_Detection` 실험을 정리했습니다. LG AI Research의 RDD 논문을 참고한 독립 실험 구현이며, 공식 구현이나 논문 전체 재현 결과는 아닙니다.

> 수식만으로 동작하는 모델은 아닙니다. Stable Diffusion v1.4의 VAE·UNet·텍스트 인코더와 LPIPS(VGG)를 사용하고, 탐지용 분류기를 추가로 학습하지 않는 **training-free** 방식입니다.

## 핵심 아이디어

이미지 복원 오차에서 블러·회전된 이미지의 오차를 빼 배경에 따른 영향을 줄이고, 잠재 공간의 회전 전후 노이즈 예측 오차를 조합합니다. 아래는 원본 노트북에서 사용한 수식입니다.

$$S_{image}(x)=f_{AE}(x)-f_{AE}(F(x))-0.5f_{AE}(R(x))+0.5f_{AE}(F(R(x)))$$

$$S_{latent}(z)=f_{latent}(R(z))-f_{latent}(z)$$

$$S_{RDD}(x)=S_{image}(x)\cdot S_{latent}(E(x))^2$$

여기서 $F$는 Gaussian blur, $R$은 90도 회전입니다. $f_{AE}$는 VAE 복원 전후 LPIPS 거리, $f_{latent}$는 노이즈 예측 MSE의 Monte Carlo 평균입니다. 원본 노트북에서는 점수가 임계값보다 크면 `real`, 이하이면 `ai_generated`로 판정했습니다. 점수는 확률이 아닙니다.

## 실행

Python 3.10 이상과 충분한 메모리가 필요합니다. GPU 사용을 권장하며 CPU도 선택할 수 있지만 오래 걸릴 수 있습니다. 첫 실행 시 Hugging Face 및 torchvision/LPIPS의 사전학습 가중치를 다운로드합니다. 모델 접근에는 해당 제공자의 접근 조건 및 모델 라이선스가 적용됩니다.

```bash
git clone https://github.com/KORTibetfox/RDD-AI-Image-Detection.git
cd RDD-AI-Image-Detection
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m rdd_detector path/to/image.jpg
```

기본 실행은 점수만 출력합니다. 원본 노트북의 실험용 임계값을 적용하려면:

```bash
python -m rdd_detector path/to/image.jpg --threshold 0.0001 --seed 42
# CPU 또는 노이즈 배치 크기 지정
python -m rdd_detector path/to/image.jpg --device cpu --noise-batch-size 1
python -m unittest discover -s tests -v
```

PyTorch와 torchvision은 플랫폼 및 CUDA 버전에 맞는 조합으로 설치하세요. `requirements.txt`는 허용 버전 범위이며, 원본 실행 환경의 lockfile은 아닙니다. CLI의 JSON 결과에는 세 점수, 선택한 임계값과 선택적 판정이 들어갑니다. 모델 로딩 메시지가 함께 출력될 수 있습니다.

## 원본에 저장된 예비 결과

아래 수치는 제공받은 노트북 출력에서 옮긴 **과거 실행 기록**이며 이번 정리 과정에서 모델을 재실행한 결과가 아닙니다.

| 실험 | 표본 수 | 원본 임계값에서 맞춘 수 | 평균 RDD |
|---|---:|---:|---:|
| Stable Diffusion v1.4로 직접 생성 | 20 | 20/20 | -0.0009518 |
| Lorem Picsum에서 수집한 Real 가정 이미지 | 20 | 20/20 | 0.0006523 |

**각 20장의 소규모 실험에서 관측된 100%이며, 일반적인 탐지 정확도 100%를 의미하지 않습니다.** 실사 표본의 독립적인 진위 검증, 검증 세트 기반 임계값 선택, 광범위한 생성 모델 평가 및 AUROC 계산은 수행하지 않았습니다. 자세한 표본 조건과 재현 한계는 [실험 기록](docs/experiments.md)에 정리했습니다.

## 파일 구성

- `rdd_detector/`: 이미지 하나를 점수화하는 모듈과 CLI. 중복 제거, 입력 검증, 시드 관리와 메모리 절약용 노이즈 분할 처리.
- `notebooks/01_rdd_walkthrough.ipynb`: 로컬 이미지로 실행하는 정리된 안내 노트북.
- `notebooks/02_original_experiment_cleaned.ipynb`: 원본 코드의 탐색 흐름 보존. 설치 로그·출력 이미지·개인 입력 경로·실행 메타데이터 제거.
- `docs/method.md`: 수식과 구현 선택, 논문 전체 재현과의 차이.
- `docs/experiments.md`: 원본 출력 수치와 검증 한계.
- `references.bib`: 참고 논문 서지 정보.

## 참고 논문 및 출처

**Sungik Choi, Hankook Lee, Jaehoon Lee, Robin Kim, Stanley Jungkyu Choi, Moontae Lee.**
*A Debiased Reconstruction-based Framework for Training-Free Detection of AI-Generated Images.* CVPR, 2026.

- [LG AI Research 공식 논문 소개](https://www.lgresearch.ai/publication/view?seq=166)
- [CVF 논문 PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Choi_A_Debiased_Reconstruction-based_Framework_for_Training-Free_Detection_of_AI-Generated_Images_CVPR_2026_paper.pdf)
- [CVF 보충자료](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Choi_A_Debiased_Reconstruction-based_CVPR_2026_supplemental.pdf)
- [원본 Kaggle 노트북](https://www.kaggle.com/code/jaewon1007/lg-detection) — 비공개 상태라면 접근이 제한될 수 있습니다.
- [Stable Diffusion v1.4 모델](https://huggingface.co/CompVis/stable-diffusion-v1-4)
- [LPIPS](https://github.com/richzhang/PerceptualSimilarity)

논문의 방법에 대한 기여는 원저자에게 있습니다. 이 저장소는 jaewon1007의 독립 실험과 그 정리본입니다. 논문 PDF, 모델 가중치 및 실험 입력 이미지는 저장소에 포함하지 않습니다. 코드에 별도 오픈소스 라이선스는 지정하지 않았으며, 공개 저장소라는 사실만으로 재배포 라이선스를 부여하는 것은 아닙니다.

## 검증 상태

정리 과정에서 Python 문법, 노트북 JSON 구조, 순수 점수 수식과 임계값 경계 테스트를 확인했습니다. 정리 환경에 PyTorch와 모델 가중치가 없어 GPU 추론 및 수정된 실행 코드의 종단간 검증은 수행하지 못했습니다. 동일 시드를 써도 환경·라이브러리·노이즈 배치 크기가 다르면 점수가 달라질 수 있습니다.

