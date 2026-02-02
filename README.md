# Project Name

## Setup

### 1. 데이터 다운로드

[Notion 링크](https://www.notion.so/250130-2f80a6d009bd803ba8cbc31b5de594d3?source=copy_link)에서 다음 파일들을 다운로드하세요:

- `character_s3_full_final_len_67566.csv` (129.4 MiB)
- `plot_s3_full_final_len_67676.csv` (114.4 MiB)

다운로드한 파일들을 `data/` 폴더에 넣어주세요:
```
project/
├── data/
│   ├── character_s3_full_final_len_67566.csv
│   └── plot_s3_full_final_len_67676.csv
├── script/
│   └── run_**.sh
└── README.md
```

### 2. 패키지 설치
```bash
pip install -r requirements.txt
```

### 3. 실행
```bash
bash script/run_**.sh
```

## Requirements

- Python 3.x
- 필요한 패키지는 `requirements.txt` 참조

