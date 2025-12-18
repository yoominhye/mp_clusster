# Materials Project Band Structure Image Downloader

Materials Project에서 밴드 구조 이미지를 자동으로 다운로드하고 처리하는 도구입니다.

## 특징

- **에너지 범위**: Fermi 에너지 기준 ±4 eV
- **이미지 크기**: 224×224 픽셀
- **이진화(Binarization)**: 흑백 이미지로 변환
- **MP-20 호환**: 단위 셀당 최대 20개 원자 필터링

이 방법론은 다음 논문을 기반으로 합니다:
> "Elf autoencoder for unsupervised exploration of flat-band materials using electronic band structure fingerprints"  
> Pentz et al., Communications Physics, 2025

---

## 설치 방법

### 1. 필수 패키지 설치

```bash
pip install -r requirements.txt
```

또는 직접 설치:

```bash
pip install mp-api pymatgen matplotlib numpy pillow tqdm
```

### 2. API 키 발급

1. [Materials Project API 페이지](https://next-gen.materialsproject.org/api) 접속
2. 로그인 또는 계정 생성
3. API 키 복사

### 3. API 키 설정

```bash
# Linux/Mac
export MP_API_KEY="your_api_key_here"

# Windows (PowerShell)
$env:MP_API_KEY="your_api_key_here"

# Windows (CMD)
set MP_API_KEY=your_api_key_here
```

---

## 사용 방법

### 방법 1: 명령줄 인터페이스 (CLI)

```bash
# 기본 사용 (MP-20 데이터셋 전체 다운로드)
python mp_bandstructure_downloader.py

# 특정 개수만 다운로드
python mp_bandstructure_downloader.py -n 100

# 특정 물질 다운로드
python mp_bandstructure_downloader.py -m mp-149 mp-13 mp-22526

# 모든 옵션 확인
python mp_bandstructure_downloader.py --help
```

### CLI 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `-k`, `--api-key` | Materials Project API 키 | 환경변수 사용 |
| `-o`, `--output-dir` | 출력 디렉토리 | `./band_structure_images` |
| `-e`, `--energy-range` | Fermi 레벨 기준 에너지 범위 (eV) | `4.0` |
| `-s`, `--image-size` | 이미지 크기 (픽셀) | `224` |
| `-a`, `--max-atoms` | 단위 셀당 최대 원자 수 | `20` |
| `-n`, `--num-materials` | 다운로드할 물질 수 | 전체 |
| `--no-binarize` | 이진화 비활성화 | False |
| `-m`, `--material-ids` | 특정 물질 ID 지정 | - |
| `-v`, `--verbose` | 상세 로그 출력 | False |

### 방법 2: Python 스크립트

```python
from mp_bandstructure_downloader import BandStructureDownloader

# 다운로더 초기화
downloader = BandStructureDownloader(
    api_key="your_api_key",      # 또는 환경변수 사용
    output_dir="./images",
    energy_range=4.0,            # ±4 eV
    image_size=(224, 224),       # 224x224 픽셀
    binarize=True                # 이진화 활성화
)

# 단일 물질 다운로드
downloader.process_single_material("mp-149", "Si")

# 여러 물질 다운로드
downloader.download_all(
    max_atoms=20,        # MP-20 기준
    num_materials=1000,  # 1000개 제한
    band_gap=(0.5, 3.0)  # 밴드갭 0.5~3.0 eV
)
```

### 방법 3: 예제 스크립트 실행

```bash
python example_usage.py
```

---

## 출력 구조

```
band_structure_images/
├── raw/                    # 원본 플롯 이미지
│   ├── mp-149_raw.png
│   ├── mp-13_raw.png
│   └── ...
├── processed/              # 처리된 이미지 (224x224, 이진화)
│   ├── mp-149.png
│   ├── mp-13.png
│   └── ...
└── metadata.json           # 메타데이터
```

### metadata.json 예시

```json
{
  "download_date": "2025-01-15T10:30:00",
  "energy_range_eV": 4.0,
  "image_size": [224, 224],
  "binarized": true,
  "total_materials": 1000,
  "successful": 985,
  "failed": 15,
  "materials": [
    {"material_id": "mp-149", "formula": "Si", "nsites": 2},
    ...
  ]
}
```

---

## 고급 사용법

### 특정 원소를 포함하는 물질 검색

```python
# Silicon과 Oxygen을 포함하는 물질
downloader.download_all(
    elements=["Si", "O"],
    max_atoms=20,
    num_materials=100
)
```

### 특정 밴드갭 범위

```python
# 반도체 범위 (0.5 ~ 3.0 eV)
downloader.download_all(
    band_gap=(0.5, 3.0),
    max_atoms=20
)
```

### 안정한 물질만 다운로드

```python
downloader.download_all(
    is_stable=True,
    max_atoms=20
)
```

---

## 주의사항

1. **API 제한**: Materials Project는 API 요청에 제한이 있습니다. 대량 다운로드 시 속도 제한에 걸릴 수 있습니다.

2. **저장 공간**: MP-20 전체 데이터셋(~45,000개)을 다운로드하면 상당한 저장 공간이 필요합니다.

3. **네트워크**: 밴드 구조 데이터는 용량이 크므로 안정적인 네트워크 연결이 필요합니다.

4. **API 키 보안**: API 키를 코드에 직접 넣지 마시고 환경변수를 사용하세요.

---

## 문제 해결

### API 키 오류
```
ValueError: API key required. Set MP_API_KEY environment variable...
```
→ 환경변수를 설정하거나 `--api-key` 옵션 사용

### 밴드 구조 없음
일부 물질은 밴드 구조 데이터가 없을 수 있습니다. 이런 경우 자동으로 건너뜁니다.

### 메모리 부족
대량 다운로드 시 `--num-materials` 옵션으로 배치 크기를 줄이세요.

---

## 라이선스

MIT License

## 참고 문헌

1. Pentz, H. K., et al. "Elf autoencoder for unsupervised exploration of flat-band materials using electronic band structure fingerprints." Communications Physics (2025).

2. Jain, A., et al. "The Materials Project: A materials genome approach to accelerating materials innovation." APL Materials 1.1 (2013): 011002.
