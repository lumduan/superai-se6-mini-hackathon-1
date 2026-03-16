# ruff: noqa: E402, F401
# ไฟล์นี้เป็น notebook-style (.py) ที่จะแปลงเป็น Jupyter Notebook
# การ import ที่อยู่หลัง cell marker และ import ที่ใช้ใน phase ถัดไปถือเป็นเรื่องปกติ

# %% [markdown]
# # 🚆 สถิติผู้โดยสารระบบขนส่งสาธารณะในประเทศไทย
# **การวิเคราะห์ปริมาณการเดินทางของประชาชนด้วยระบบขนส่งสาธารณะ**
#
# **ชุดข้อมูล:** สถิติผู้โดยสารรายวันจากระบบขนส่งสาธารณะทั่วประเทศไทย (ปี 2568–2569)
# **ระยะเวลา:** ประมาณ 14 เดือน
# **แหล่งที่มา:** กระทรวงคมนาคม (Ministry of Transport)
#
# ---
#
# ## วัตถุประสงค์การวิเคราะห์
# 1. **สัดส่วนการใช้ระบบขนส่ง (Modal Share)** — ระบบขนส่งใดมีผู้โดยสารมากที่สุด?
# 2. **เปรียบเทียบรถไฟฟ้าในเมือง** — รูปแบบการเดินทางของแต่ละสายแตกต่างกันอย่างไร?
# 3. **ตรวจจับเหตุการณ์พิเศษ** — สามารถตรวจพบวันหยุดและเทศกาลจากข้อมูลได้หรือไม่?
# 4. **พยากรณ์ผู้โดยสาร** — ทำนายปริมาณผู้โดยสารล่วงหน้า 30 วัน ด้วย Facebook Prophet

# %% [markdown]
# ---
# ## ⚙️ ติดตั้ง Dependencies

# %%
# ติดตั้งแพ็คเกจที่จำเป็น (รันครั้งเดียวใน Colab)
import subprocess
subprocess.run(['pip', 'install', 'prophet', 'plotly', 'scikit-learn', '-q'], check=True)

# %%
# นำเข้าไลบรารีทั้งหมดที่ใช้ในการวิเคราะห์
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
import warnings
warnings.filterwarnings('ignore')

print('✅ นำเข้าไลบรารีทั้งหมดสำเร็จ')

# %% [markdown]
# ---
# ## Phase 1 — โหลดข้อมูล (Data Loading)

# %%
# กำหนด URL ของชุดข้อมูลผู้โดยสารปี 2568 และ 2569 จาก GitHub
URL_68 = 'https://raw.githubusercontent.com/lumduan/superai-se6-mini-hackathon-1/main/dataset-5/data/passengers68.csv'
URL_69 = 'https://raw.githubusercontent.com/lumduan/superai-se6-mini-hackathon-1/main/dataset-5/data/passengers69.csv'


def load_dataset(url: str, name: str) -> pd.DataFrame:
    """โหลดชุดข้อมูลจาก URL พร้อมตรวจจับข้อผิดพลาด"""
    try:
        df = pd.read_csv(url)
        print(f'✅ โหลด {name} สำเร็จ: {df.shape}')
        return df
    except Exception as e:
        print(f'❌ โหลด {name} ล้มเหลว')
        print(e)
        raise


# โหลดชุดข้อมูลแต่ละปีแยกกัน — หากไฟล์ใดเสียหายจะรู้ได้ทันที
df68 = load_dataset(URL_68, 'passengers68')
df69 = load_dataset(URL_69, 'passengers69')

# %%
# ตรวจสอบความสอดคล้องของ schema ก่อนรวมข้อมูล
print('ตรวจสอบความสอดคล้องของคอลัมน์...')
assert set(df68.columns) == set(df69.columns), 'คอลัมน์ของชุดข้อมูลสองปีไม่ตรงกัน กรุณาตรวจสอบ'

print('คอลัมน์ passengers68:', df68.columns.tolist())
print('คอลัมน์ passengers69:', df69.columns.tolist())

# %%
# รวมชุดข้อมูลทั้งสองปีเข้าด้วยกัน และตรวจสอบว่าผลลัพธ์ไม่ว่างเปล่า
df = pd.concat([df68, df69], ignore_index=True)

assert not df.empty, 'ชุดข้อมูลว่างเปล่า กรุณาตรวจสอบไฟล์ต้นฉบับ'

print(f'\nขนาดชุดข้อมูลรวม: {df.shape}')
print(f'จำนวนแถว: {len(df):,}')
print(f'จำนวนคอลัมน์: {len(df.columns)}')

# แสดงข้อมูลสรุปประเภทและค่า null ของแต่ละคอลัมน์
df.info()

# %%
# แสดงตัวอย่างข้อมูล 5 แถวแรก
df.head()

# %%
# แสดงตัวอย่างข้อมูลแบบสุ่ม 5 แถว เพื่อดูความหลากหลายของข้อมูล
df.sample(5, random_state=42)

# %%
# ตรวจสอบโครงสร้างของข้อมูล — ค่าที่ไม่ซ้ำในคอลัมน์สำคัญ
print('รูปแบบการเดินทางที่มีในข้อมูล (รูปแบบการเดินทาง):')
print(df['รูปแบบการเดินทาง'].unique())
print('\nประเภทยานพาหนะ/ท่าที่มีในข้อมูล (ยานพาหนะ/ท่า):')
print(df['ยานพาหนะ/ท่า'].unique())
