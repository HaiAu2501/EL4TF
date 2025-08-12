import pandas as pd
from pathlib import Path

def _process_file(symbol: str, folder: str = 'regression') -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(__file__).resolve().parent.parent / 'data' / 'vn30' / folder
    df_train = pd.read_csv(data_dir / f'{symbol}_train.csv', parse_dates=['time'])
    df_test = pd.read_csv(data_dir / f'{symbol}_test.csv', parse_dates=['time'])
    return df_train, df_test

VN30 = [
    'ACB', 'BCM', 'BID', 'BVH', 'CTG', 
    'FPT', 'GAS', 'GVR', 'HDB', 'HPG', 
    'LPB', 'MBB', 'MSN', 'MWG', 'PLX',
    'SAB', 'SHB', 'SSB', 'SSI', 'STB',
    'TCB', 'TPB', 'VCB', 'VHM', 'VIB',
    'VIC', 'VJC', 'VNM', 'VPB', 'VRE',
]

TARGETS = ['open', 'high', 'low', 'close'] # What we want to predict

# Example
# if __name__ == "__main__":
#     _process_file('ACB', folder='regression')