import pandas as pd
import os

def load_and_preprocess(file_path, count=None, seed=42, run_type='test'):
    """
    파일을 로드하고 샘플링 수행
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

    print(f"Loading Dataset...: {file_path}")
    try:
        if run_type == 'test':
            df = pd.read_csv(file_path).head(500)
        else: 
            df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"CSV 읽기 실패: {e}")

    # Data Sampling 
    group_cols = ['Source', 'Category', 'ID', 'Version']

    sampled_df = df.groupby(group_cols, group_keys=False).apply(
        lambda x: x.sample(n=count, random_state=seed)
    )

    sampled_df = sampled_df.reset_index(drop=True)

    return sampled_df