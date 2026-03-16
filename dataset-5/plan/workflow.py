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
subprocess.run(['uv', 'pip', 'install', 'prophet', 'plotly', 'scikit-learn', '-q'], check=True)

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

# %% [markdown]
# ---
# ## Phase 2 — ทำความสะอาดและตรวจสอบข้อมูล (Data Cleaning & Validation)
#
# หลักการ: **clean → fix → fill → drop (ขั้นสุดท้ายเท่านั้น)**
# การลบแถวทันทีอาจทำให้เสีย time-series signal โดยไม่จำเป็น

# %% [markdown]
# ### 2.1 ลบแถวที่ว่างเปล่าทั้งหมด
# ลบเฉพาะแถวที่ทุกคอลัมน์เป็น NaN — เกิดจาก Excel formatting ในข้อมูลภาครัฐ

# %%
rows_before = len(df)
df = df.dropna(how='all').reset_index(drop=True)
print(f'แถวก่อนลบแถวว่าง: {rows_before:,}')
print(f'แถวหลังลบแถวว่าง:  {len(df):,}  (ลบ {rows_before - len(df):,} แถว)')

# %% [markdown]
# ### 2.2 กรองเฉพาะหน่วย "คน" (Enforce Unit Consistency)
# ชุดข้อมูลนี้มีทั้ง "คน" (ผู้โดยสาร) และ "คัน" (ยานพาหนะ)
# เก็บเฉพาะ "คน" เพื่อให้ตัวเลขมีความหมายเดียวกันตลอดการวิเคราะห์

# %%
print('หน่วยที่มีในข้อมูล:')
print(df['หน่วย'].value_counts())

# กรองเฉพาะหน่วย "คน" — ลบแถว "คัน" ออก
df = df[df['หน่วย'] == 'คน'].reset_index(drop=True)
print(f'\nแถวหลังกรองเฉพาะหน่วย "คน": {len(df):,}')

# %% [markdown]
# ### 2.3 แปลงรูปแบบวันที่ (Fix Date Format)
# ใช้ `errors='coerce'` — วันที่ผิดรูปแบบกลายเป็น NaT แทนการ crash
# จากนั้น enforce ช่วงปีที่ถูกต้อง (2025–2026)

# %%
# แปลงวันที่จาก string (DD/MM/YYYY) → datetime
df['date'] = pd.to_datetime(df['วันที่'], dayfirst=True, errors='coerce')

bad_date = df['date'].isna()
print(f'แถวที่แปลงวันที่ไม่ได้ (NaT): {bad_date.sum():,}')
if bad_date.sum() > 0:
    print(df[bad_date][['วันที่']].head())
df = df[~bad_date].reset_index(drop=True)

# %%
# enforce ช่วงปีที่ถูกต้อง (2025–2026) — ป้องกัน outlier จากการกรอกผิด
print('การกระจายของปีในข้อมูล:')
print(df['date'].dt.year.value_counts().sort_index())

invalid_year = ~df['date'].dt.year.between(2025, 2026)
print(f'\nแถวที่ปีอยู่นอกช่วง 2025–2026: {invalid_year.sum():,}')
if invalid_year.sum() > 0:
    print(df[invalid_year][['date', 'ยานพาหนะ/ท่า']].head())
df = df[~invalid_year].reset_index(drop=True)

# เรียงลำดับตาม ยานพาหนะ/ท่า + วันที่ — จำเป็นก่อน interpolate
df = df.sort_values(['ยานพาหนะ/ท่า', 'date']).reset_index(drop=True)
print(f'\nช่วงวันที่: {df["date"].min().date()} → {df["date"].max().date()}')
print(f'จำนวนวันในช่วง: {(df["date"].max() - df["date"].min()).days + 1} วัน')

# %% [markdown]
# ### 2.4 แปลงและซ่อมคอลัมน์ปริมาณ (Fix Passenger Column)
# ลำดับ: ลบ comma → แปลง numeric → interpolate ภายในกลุ่มยานพาหนะ → drop ที่ซ่อมไม่ได้
#
# **สำคัญ:** ต้องเรียงตาม `[ยานพาหนะ/ท่า, date]` ก่อน interpolate เสมอ
# เพราะ interpolate ใช้ลำดับของ row — ถ้าไม่เรียงจะ fill ข้ามประเภทขนส่ง (BTS → MRT)

# %%
# ขั้น 1: ลบ comma และ whitespace (เช่น "1,200" → "1200")
df['ปริมาณ'] = df['ปริมาณ'].astype(str).str.replace(',', '', regex=False).str.strip()

# ขั้น 2: แปลงเป็น numeric — ค่าที่แปลงไม่ได้ (เช่น "-", "N/A") กลายเป็น NaN
df['ปริมาณ'] = pd.to_numeric(df['ปริมาณ'], errors='coerce')
print(f'ประเภทข้อมูล ปริมาณ: {df["ปริมาณ"].dtype}  |  NaN: {df["ปริมาณ"].isna().sum():,}')

# %%
# ขั้น 3: interpolate ภายในแต่ละกลุ่มยานพาหนะ/ท่า
# limit=3 ป้องกันการ fill ช่องว่างยาวเกินไป ซึ่งจะบิดเบือน weekly pattern
nan_before = df['ปริมาณ'].isna().sum()

df['ปริมาณ'] = (
    df.groupby('ยานพาหนะ/ท่า')['ปริมาณ']
    .transform(lambda x: x.interpolate(method='linear', limit=3))
)

nan_after = df['ปริมาณ'].isna().sum()
print(f'NaN ก่อน interpolate: {nan_before:,}  →  หลัง: {nan_after:,}  (fill: {nan_before - nan_after:,})')

# %%
# ขั้น 4 (final drop): ลบแถวที่ยัง NaN หลัง interpolate — ซ่อมไม่ได้จริงๆ
df = df[df['ปริมาณ'].notna()].reset_index(drop=True)
print(f'แถวสุดท้ายหลังทำความสะอาด: {len(df):,}')

# %% [markdown]
# ### 2.5 รายงานคุณภาพข้อมูล (Data Quality Report)

# %%
print('=' * 50)
print('รายงานคุณภาพข้อมูล (Data Quality Report)')
print('=' * 50)

# 1. ค่า Missing ในแต่ละคอลัมน์
print('\n1. ค่า Missing ในแต่ละคอลัมน์:')
print(df.isna().sum())

# %%
# 2. ตรวจสอบแถวซ้ำด้วย domain key: date + ยานพาหนะ/ท่า
# (ถูกต้องกว่า df.duplicated() เพราะ primary key ของชุดข้อมูลนี้คือคู่นี้)
dup = df[df.duplicated(subset=['date', 'ยานพาหนะ/ท่า'])]
print(f'\n2. แถวที่ซ้ำกัน (date + ยานพาหนะ/ท่า): {len(dup):,}')
if len(dup) > 0:
    print(dup[['date', 'ยานพาหนะ/ท่า', 'ปริมาณ']].head())
df = df.drop_duplicates(subset=['date', 'ยานพาหนะ/ท่า']).reset_index(drop=True)
print(f'   แถวหลังลบซ้ำ: {len(df):,}')

# %%
# 3. ตรวจสอบค่าปริมาณติดลบ (ผู้โดยสารไม่มีทางติดลบ)
neg_mask = df['ปริมาณ'] < 0
print(f'\n3. แถวที่มีปริมาณติดลบ: {neg_mask.sum():,}')
if neg_mask.sum() > 0:
    print(df[neg_mask][['date', 'ยานพาหนะ/ท่า', 'ปริมาณ']].head())
df = df[~neg_mask].reset_index(drop=True)

# %%
# 4. สถิติเชิงพรรณนา — min>=0 | max ไม่สูงกว่า mean มาก | std>>mean บ่งบอก outlier
print('\n4. สถิติเชิงพรรณนาของ ปริมาณ:')
print(df['ปริมาณ'].describe().round(2))

# %% [markdown]
# ### 2.6 Sanity Checks — ตรวจสอบความสมเหตุสมผลของข้อมูล

# %%
# 5. ตรวจสอบความสม่ำเสมอของชื่อยานพาหนะ/ท่า
# ข้อมูลภาครัฐบางครั้งมีชื่อเดียวกันแต่เขียนต่างกัน เช่น "BTS" vs "BTS-Green Line"
print('5. ตรวจสอบชื่อยานพาหนะ/ท่า:')
print(f'   จำนวนประเภทที่ไม่ซ้ำ: {df["ยานพาหนะ/ท่า"].nunique()}')
print(df['ยานพาหนะ/ท่า'].unique())

# %%
# 6. ตรวจสอบ time-series gap ระดับ dataset
print('\n6. ตรวจสอบ gap ของวันที่ (dataset level):')
date_range_full = pd.date_range(df['date'].min(), df['date'].max())
missing_dates = date_range_full.difference(df['date'])
print(f'   จำนวนวันที่ไม่ซ้ำ: {df["date"].nunique()}  |  วันที่หายไป: {len(missing_dates)}')
if len(missing_dates) > 0:
    print(f'   วันที่หายไป (5 แรก): {missing_dates[:5].tolist()}')

# %%
# ตรวจสอบ gap ระดับยานพาหนะ/ท่า — ตรวจจับสายที่มีข้อมูลขาดหาย
print('\n   gap ระดับยานพาหนะ/ท่า:')
gap_found = False
for name, group in df.groupby('ยานพาหนะ/ท่า'):
    full = pd.date_range(group['date'].min(), group['date'].max())
    gap = len(full.difference(group['date']))
    if gap > 0:
        print(f'   {name}: {gap} วันที่หายไป')
        gap_found = True
if not gap_found:
    print('   ทุกสายมีข้อมูลครบถ้วน ✅')

# %%
# 7. ตรวจสอบ extreme outlier ด้วย percentile 99
q99 = df['ปริมาณ'].quantile(0.99)
print(f'\n7. ค่า 99th percentile: {q99:,.0f}')
print('แถวที่ปริมาณเกิน 99th percentile:')
print(df[df['ปริมาณ'] > q99][['date', 'ยานพาหนะ/ท่า', 'ปริมาณ']].head(10).to_string(index=False))

# %%
# 8. สถิติปริมาณแยกตามยานพาหนะ/ท่า — เพื่อดูสัดส่วนและความสมเหตุสมผล
print('\n8. สถิติปริมาณแยกตามยานพาหนะ/ท่า:')
print(df.groupby('ยานพาหนะ/ท่า')['ปริมาณ'].describe().round(0).to_string())

# %% [markdown]
# ### 2.7 กรองเฉพาะข้อมูลการขนส่งทางราง (Filter Rail Transport)
# ต้องกรองก่อน pivot เสมอ เพื่อป้องกันข้อมูลทางถนน/ทางน้ำ/ทางอากาศปะปนเข้ามา

# %%
# กรองข้อมูลเฉพาะ: ทางราง + สาธารณะ เท่านั้น
rail_df = df[
    (df['รูปแบบการเดินทาง'] == 'ทางราง') &
    (df['สาธารณะ/ส่วนบุคคล'] == 'สาธารณะ')
].copy()

print(f'แถวข้อมูลทางราง: {len(rail_df):,}  (จากทั้งหมด: {len(df):,})')
print('\nสาย/ประเภทยานพาหนะในระบบราง:')
print(rail_df['ยานพาหนะ/ท่า'].value_counts())

# %%
# สรุปช่วงวันที่และความครบถ้วนของข้อมูลทางราง
print(f'\nช่วงวันที่ข้อมูลทางราง: {rail_df["date"].min().date()} → {rail_df["date"].max().date()}')
print(f'จำนวนวันที่ไม่ซ้ำ: {rail_df["date"].nunique()} วัน')

# %% [markdown]
# ---
# ## Phase 3 — แปลงรูปแบบข้อมูล (Data Transformation)
#
# เปลี่ยนจาก **long format** (date | transport | passengers)
# เป็น **wide format** (date × rail line) เพื่อให้ Prophet และ correlation analysis ใช้งานได้

# %% [markdown]
# ### 3.1 แปลงชื่อยานพาหนะจากภาษาไทยเป็นภาษาอังกฤษ
# กำหนด mapping เฉพาะ **7 สายรถไฟฟ้าในเมือง** ที่เป็นเป้าหมายของการวิเคราะห์
# ไม่รวมรถไฟแห่งชาติ (รถไฟ, รถไฟ ขาเข้า/ขาออกประเทศ) เพราะเป็นระบบต่างประเภท

# %%
# mapping ชื่อยานพาหนะภาษาไทย → ชื่อคอลัมน์ภาษาอังกฤษ
vehicle_map = {
    'รถไฟฟ้า BTS':         'BTS',
    'รถไฟฟ้าสายสีน้ำเงิน': 'MRT Blue',
    'รถไฟฟ้าสายสีม่วง':    'MRT Purple',
    'รถไฟฟ้าสายสีเหลือง':  'MRT Yellow',
    'รถไฟฟ้าสายสีชมพู':    'MRT Pink',
    'รถไฟฟ้า ARL':         'Airport Rail Link',
    'รถไฟฟ้าสายสีแดง':     'SRT Red',
}

# แปลงชื่อยานพาหนะ — ค่าที่ไม่อยู่ใน mapping (รถไฟแห่งชาติ) กลายเป็น NaN
rail_df = rail_df.copy()
rail_df['line'] = rail_df['ยานพาหนะ/ท่า'].map(vehicle_map)

# แสดงรายการยานพาหนะที่ตัดออก เพื่อความโปร่งใสของ pipeline
unmapped = rail_df[rail_df['line'].isna()]['ยานพาหนะ/ท่า'].unique()
print('ยานพาหนะที่ตัดออกจากการวิเคราะห์ (ไม่ใช่ urban rail):')
for v in unmapped:
    print(f'  - {v}')

# กรองเฉพาะ 7 สายรถไฟฟ้าในเมืองที่ map แล้ว
rail_df = rail_df[rail_df['line'].notna()].copy()

# ตรวจสอบว่ายังมีข้อมูลเหลืออยู่ — หาก vehicle_map ผิดพลาดจะรู้ได้ทันที
assert len(rail_df) > 0, 'ไม่มีข้อมูล urban rail หลัง mapping — ตรวจสอบ vehicle_map'

print(f'\nแถวหลังกรองเฉพาะ urban rail: {len(rail_df):,}')
print('สายที่เหลือ:', sorted(rail_df['line'].unique()))

# %% [markdown]
# ### 3.2 Pivot จาก Long Format เป็น Wide Format
# แกน index = `date` | คอลัมน์ = สาย | ค่า = ปริมาณผู้โดยสาร
# ใช้ `aggfunc='sum'` เพื่อรองรับกรณีที่มีหลายแถวต่อวันต่อสาย

# %%
# ตรวจสอบ duplicate (date + line) ก่อน pivot — แสดงให้เห็น data quality ก่อน aggregate
dup_rail = rail_df.duplicated(subset=['date', 'line'], keep=False)
print(f'แถวที่ซ้ำ (date + line) ก่อน pivot: {dup_rail.sum():,}')
if dup_rail.sum() > 0:
    print('ตัวอย่าง (pivot_table จะ sum ค่าเหล่านี้เข้าด้วยกัน):')
    print(rail_df[dup_rail][['date', 'line', 'ปริมาณ']].head())

# %%
# เรียงลำดับตามวันที่ก่อน pivot เพื่อให้ pipeline ชัดเจนและ deterministic
rail_df = rail_df.sort_values('date').reset_index(drop=True)

# pivot_table พร้อม aggfunc='sum' รองรับกรณีมีหลายแถวต่อวันต่อสาย
pivot_df = rail_df.pivot_table(
    index='date',
    columns='line',
    values='ปริมาณ',
    aggfunc='sum',
)
pivot_df.columns.name = None  # ลบ label "line" ออกจาก column axis

# เรียงลำดับคอลัมน์ให้ deterministic ทุกครั้งที่รัน
pivot_df = pivot_df.sort_index(axis=1)

print(f'รูปแบบหลัง pivot: {pivot_df.shape}  (แถว=วัน, คอลัมน์=สาย)')
print('คอลัมน์:', pivot_df.columns.tolist())

# %%
# ตรวจสอบ missing values ก่อน fill — แสดง pattern ก่อนแก้ไข
print('Missing values ต่อสายก่อน fill:')
print(pivot_df.isna().sum().sort_values(ascending=False))

# %% [markdown]
# ### 3.3 เติมวันที่ที่หายไป (Handle Missing Dates)
# Prophet ต้องการ time series ที่ต่อเนื่องทุกวัน — ห้ามมี gap
#
# กลยุทธ์การเติมข้อมูล:
# | สถานการณ์ | กลยุทธ์ |
# |---|---|
# | ข้อมูลขาดหาย ≤3 วัน (บันทึกไม่ครบ) | `interpolate(method='time', limit_area='inside')` |
# | ข้อมูลขาดหายนานกว่านั้น (หยุดให้บริการ) | `fillna(0)` |

# %%
# ตรวจสอบก่อนว่า index เป็น datetime — asfreq() ต้องการ DatetimeIndex เท่านั้น
assert pd.api.types.is_datetime64_any_dtype(pivot_df.index), \
    'pivot_df index ไม่ใช่ DatetimeIndex — ตรวจสอบขั้นตอน date conversion'

# เรียงลำดับ index ก่อน reindex เพื่อความปลอดภัย
pivot_df = pivot_df.sort_index()

# reindex ให้ครบทุกวัน — วันที่ขาดหายจะกลายเป็น NaN
pivot_df = pivot_df.asfreq('D')
print(f'รูปแบบหลัง reindex: {pivot_df.shape}')

# %%
# interpolate ช่องว่างสั้น ≤3 วัน (data recording gap)
# limit_area='inside' ป้องกันการ fill ที่ปลาย series
# (เช่น NaN NaN 200 → ไม่ fill ด้านหน้า แต่ 100 NaN 200 → fill ตรงกลาง)
pivot_df = pivot_df.interpolate(
    method='time',
    limit=3,
    limit_area='inside',
)

# zero-fill ช่องว่างที่เหลือ (>3 วัน หรือปลาย series = หยุดให้บริการ)
pivot_df = pivot_df.fillna(0)

print(f'NaN หลัง fill ทั้งหมด: {pivot_df.isna().sum().sum()}  (ควรเป็น 0 ✅)')

# %% [markdown]
# ### 3.4 ตรวจสอบความสมบูรณ์ของช่วงวันที่ (Date Range Integrity Check)
# ยืนยัน index เรียงต่อเนื่องและไม่มี gap ก่อนส่งต่อ Phase 4–8

# %%
assert pivot_df.index.is_monotonic_increasing, 'Date index ไม่ได้เรียงลำดับ!'

expected_range = pd.date_range(
    start=pivot_df.index.min(),
    end=pivot_df.index.max(),
    freq='D',
)
missing_dates = expected_range.difference(pivot_df.index)

if len(missing_dates) == 0:
    print(f'✅ Date range สมบูรณ์: {pivot_df.index.min().date()} → {pivot_df.index.max().date()}')
    print(f'   จำนวนวันทั้งหมด: {len(pivot_df):,} วัน')
else:
    print(f'⚠️ พบวันที่หายไป {len(missing_dates)} วัน:', missing_dates.tolist())

# %% [markdown]
# ### 3.5 Feature Engineering
# เพิ่ม time features สำหรับ EDA, Event Detection และ Prophet regressors

# %%
# เพิ่ม time features
pivot_df['year']  = pivot_df.index.year
pivot_df['month'] = pivot_df.index.month

# day_of_week เป็น Categorical (ordered) เพื่อให้ plot/groupby เรียงลำดับถูกต้องเสมอ
pivot_df['day_of_week'] = pd.Categorical(
    pivot_df.index.day_name(),
    categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
    ordered=True,
)

# dow เป็นตัวเลข 0–6 (0=Monday) — สะดวกสำหรับ model ที่รับเฉพาะ numeric input
pivot_df['dow'] = pivot_df.index.weekday

# is_weekend เป็น int (0/1) — Prophet และ ML model ชอบ numeric มากกว่า bool
pivot_df['is_weekend'] = (pivot_df.index.weekday >= 5).astype(int)

# %%
# กำหนดรายชื่อสายรถไฟฟ้า (single authoritative list สำหรับ Phase 4–8 ทั้งหมด)
rail_lines = [col for col in [
    'BTS',
    'MRT Blue', 'MRT Purple', 'MRT Yellow', 'MRT Pink',
    'Airport Rail Link',
    'SRT Red',
] if col in pivot_df.columns]

# ตรวจสอบว่ามีสายรถไฟฟ้าครบตามที่คาดหวัง
assert len(rail_lines) >= 5, \
    f'พบสายรถไฟฟ้าน้อยกว่าที่คาดไว้ ({len(rail_lines)} สาย) — ตรวจสอบ vehicle_map'

# total_passengers รวมเฉพาะ rail lines แล้ว round เป็น int (ค่าหลัง interpolate อาจเป็น float)
pivot_df['total_passengers'] = pivot_df[rail_lines].sum(axis=1).round().astype(int)

print(f'rail_lines ที่ใช้วิเคราะห์ ({len(rail_lines)} สาย): {rail_lines}')
print(f'คอลัมน์ทั้งหมดใน pivot_df: {pivot_df.columns.tolist()}')

# %%
# แสดงตัวอย่างข้อมูล 7 วันแรกหลัง transformation (1 สัปดาห์)
pivot_df[rail_lines + ['total_passengers', 'year', 'month', 'day_of_week', 'is_weekend']].head(7)

# %% [markdown]
# ---
# ## Phase 4 — วิเคราะห์สัดส่วนการใช้ระบบขนส่ง (Modal Share Analysis)
# ### Challenge 1: คนไทยเดินทางด้วยอะไรมากที่สุด?
#
# จัดกลุ่มสายรถไฟฟ้าเป็น 4 ระบบหลัก:
# | ระบบ | สาย |
# |------|-----|
# | BTS  | BTS |
# | MRT  | MRT Blue + Purple + Yellow + Pink |
# | ARL  | Airport Rail Link |
# | SRT  | SRT Red |

# %% [markdown]
# ### 4.1 คำนวณ Modal Share

# %%
# กำหนดกลุ่มสายรถไฟฟ้าแต่ละระบบ (กรองเฉพาะสายที่มีอยู่ใน pivot_df)
modal_cols = {
    'BTS': [c for c in ['BTS'] if c in pivot_df.columns],
    'MRT': [c for c in ['MRT Blue', 'MRT Purple', 'MRT Yellow', 'MRT Pink'] if c in pivot_df.columns],
    'ARL': [c for c in ['Airport Rail Link'] if c in pivot_df.columns],
    'SRT': [c for c in ['SRT Red'] if c in pivot_df.columns],
}
# ตัด mode ที่ไม่มีข้อมูล — ป้องกัน KeyError ถ้า dataset เปลี่ยน schema
modal_cols = {k: v for k, v in modal_cols.items() if v}

print('กลุ่มสายรถไฟฟ้าแต่ละระบบ:')
for mode, cols in modal_cols.items():
    print(f'  {mode}: {cols}')

# %%
# คำนวณผู้โดยสารรวมและ % share ของแต่ละระบบ
modal_total = pd.Series({
    mode: pivot_df[cols].sum().sum()
    for mode, cols in modal_cols.items()
})

modal_share = (modal_total / modal_total.sum() * 100).round(2)

share_df = pd.DataFrame({
    'mode':             modal_total.index,
    'total_passengers': modal_total.values.astype(int),
    'share_pct':        modal_share.values,
}).sort_values('total_passengers', ascending=False).reset_index(drop=True)

print('\nสัดส่วนการใช้ระบบขนส่ง:')
print(share_df.to_string(index=False))

# %% [markdown]
# ### 4.2 กราฟที่ 1 — Pie Chart สัดส่วนผู้โดยสารแต่ละระบบ

# %%
# Chart 1: Modal Share Pie Chart (donut)
# ใช้ total_passengers จริง — Plotly คำนวณ % เอง แม่นยำกว่าใช้ share_pct
fig = px.pie(
    share_df,
    values='total_passengers',
    names='mode',
    title='สัดส่วนการใช้ระบบขนส่งทางราง (2025–2026)',
    hole=0.4,
    color_discrete_sequence=px.colors.qualitative.Set2,
)
fig.update_traces(
    textposition='outside',
    textinfo='percent+label',
    pull=[0.02] * len(share_df),  # เว้นระยะแต่ละชิ้นเล็กน้อยเพื่อให้ label ไม่ชนกัน
)
fig.update_layout(showlegend=True)
fig.show()

# %% [markdown]
# ### 4.3 กราฟที่ 2 — Stacked Area Chart สัดส่วนรายวัน

# %%
# คำนวณ modal share รายวัน (% ต่อวัน) เพื่อดูการเปลี่ยนแปลงตลอด 14 เดือน
modal_daily = pd.DataFrame(
    {mode: pivot_df[cols].sum(axis=1) for mode, cols in modal_cols.items()},
    index=pivot_df.index,
)
modal_daily.index.name = 'date'  # กำหนด index name เพื่อให้ reset_index().melt() ทำงานถูกต้อง

# หาร % share รายวัน — replace(0, pd.NA) ป้องกัน division by zero วันที่ไม่มีบริการ
daily_sum = modal_daily.sum(axis=1).replace(0, pd.NA)
modal_share_daily = (
    modal_daily.div(daily_sum, axis=0)
    .multiply(100)
    .round(2)       # ลด floating noise ให้กราฟเรียบขึ้น
    .sort_index()   # เรียงตามวันที่ก่อน plot เพื่อป้องกัน shuffle
)

# Chart 2: Modal Share Stacked Area Over Time
fig = px.area(
    modal_share_daily.reset_index().melt(
        id_vars='date',
        var_name='ระบบขนส่ง',
        value_name='สัดส่วน (%)',
    ),
    x='date',
    y='สัดส่วน (%)',
    color='ระบบขนส่ง',
    title='สัดส่วนการใช้ระบบขนส่งรายวัน (%)',
    color_discrete_sequence=px.colors.qualitative.Set2,
    labels={'date': 'วันที่'},
)
fig.update_layout(yaxis_range=[0, 100])
fig.show()

# %% [markdown]
# ### 4.4 กราฟที่ 3 — YoY Growth Bar Chart (2025 → 2026)
# สูตร: YoY Growth (%) = (เฉลี่ยต่อวัน₂₀₂₆ − เฉลี่ยต่อวัน₂₀₂₅) / เฉลี่ยต่อวัน₂₀₂₅ × 100
#
# **สำคัญ:** ใช้ **ค่าเฉลี่ยต่อวัน** แทนผลรวมรวม เพราะ 2026 มีข้อมูลเพียงบางเดือน
# การใช้ total sum จะทำให้ 2026 ดูเล็กกว่าความเป็นจริงเสมอ

# %%
# คำนวณผู้โดยสาร **เฉลี่ยต่อวัน** ต่อระบบ ต่อปี (robust — เฉพาะ columns ที่มีอยู่จริง)
all_modal_cols  = sum(modal_cols.values(), [])
yearly_days     = pivot_df.groupby('year').size()  # จำนวนวันที่มีในแต่ละปี
yearly_base_sum = pivot_df.groupby('year')[all_modal_cols].sum()

# เฉลี่ยต่อวัน = ผลรวมต่อปี / จำนวนวันในปีนั้น
yearly_base_avg = yearly_base_sum.div(yearly_days, axis=0)

yearly_modal = pd.DataFrame({
    mode: yearly_base_avg[cols].sum(axis=1)
    for mode, cols in modal_cols.items()
})

available_years = yearly_modal.index.tolist()
print('ปีที่มีในข้อมูล:', available_years)
print(f'จำนวนวันต่อปี:\n{yearly_days.to_string()}')

# %%
# คำนวณ YoY growth เฉพาะเมื่อมีข้อมูลทั้ง 2 ปี
if 2025 in available_years and 2026 in available_years:
    growth = ((yearly_modal.loc[2026] - yearly_modal.loc[2025]) / yearly_modal.loc[2025]) * 100
    growth_df = growth.reset_index()
    growth_df.columns = ['mode', 'yoy_growth_pct']
    growth_df['yoy_growth_pct'] = growth_df['yoy_growth_pct'].round(2)
    growth_df = growth_df.sort_values('yoy_growth_pct', ascending=False).reset_index(drop=True)

    print('\nYoY Growth (เฉลี่ยต่อวัน 2025 → 2026):')
    print(growth_df.to_string(index=False))

    # Chart 3: YoY Growth Bar Chart
    fig = px.bar(
        growth_df,
        x='mode',
        y='yoy_growth_pct',
        color='yoy_growth_pct',
        color_continuous_scale='RdYlGn',
        title='การเติบโตของผู้โดยสารเทียบปีต่อปี (YoY Growth — เฉลี่ยต่อวัน 2025 → 2026)',
        labels={'yoy_growth_pct': 'Growth (%)', 'mode': 'ระบบขนส่ง'},
        text='yoy_growth_pct',
    )
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.add_hline(y=0, line_dash='dash', line_color='gray')
    fig.show()
else:
    print('⚠️ ต้องการข้อมูลทั้งปี 2025 และ 2026 เพื่อคำนวณ YoY Growth')
    print(f'   ปีที่มี: {available_years}')

# %% [markdown]
# ### 4.5 สรุปข้อค้นพบ (Modal Share Insights)

# %%
# สรุปเชิงปริมาณ — ใช้ใน Phase 9 (Insights & Conclusion)
top_mode       = share_df.iloc[0]
avg_daily_pass = modal_total.sum() / len(pivot_df)

print('=== Modal Share Summary ===')
print(f'ระบบขนส่งที่มีผู้โดยสารมากที่สุด: {top_mode["mode"]} ({top_mode["share_pct"]:.1f}%)')
print(f'ผู้โดยสารรวมตลอด {len(pivot_df)} วัน:  {int(modal_total.sum()):,} คน')
print(f'เฉลี่ยต่อวัน (ทุกระบบรวม):          {avg_daily_pass:,.0f} คน/วัน')

if 2025 in available_years and 2026 in available_years:
    fastest_growth = growth_df.iloc[0]
    most_decline   = growth_df.iloc[-1]
    print(f'\nระบบที่เติบโตเร็วที่สุด:  {fastest_growth["mode"]} ({fastest_growth["yoy_growth_pct"]:+.1f}%)')
    print(f'ระบบที่หดตัวมากที่สุด:    {most_decline["mode"]} ({most_decline["yoy_growth_pct"]:+.1f}%)')

# %% [markdown]
# ---
# ## Phase 5 — เปรียบเทียบระบบรถไฟฟ้าในเมือง (Urban Rail Comparison)
#
# เปรียบเทียบพฤติกรรมผู้โดยสารระหว่างสายรถไฟฟ้าแต่ละสาย
# สายที่วิเคราะห์: BTS, MRT Blue, MRT Purple, MRT Yellow, MRT Pink, Airport Rail Link, SRT Red
#
# **การวิเคราะห์ใน Phase นี้:**
# 1.  Time Series — ผู้โดยสารรายวันแต่ละสาย
# 2.  Normalized Index — เปรียบเทียบการเติบโตบน Scale เดียวกัน
# 3.  Average Ridership Ranking — จัดอันดับค่าเฉลี่ย
# 4.  Volatility CV — ความผันผวนสัมพัทธ์
# 5.  Rolling 30-Day Std — ความผันผวนตามเวลา (ไม่รวมวันที่ไม่มีบริการ)
# 6.  Weekday Seasonality — รูปแบบรายวัน (Raw + Normalized)
# 7.  Rolling 30-Day YoY Growth — แนวโน้มการเติบโต
# 8.  Clustered Correlation Heatmap (Log-transform) — ความสัมพันธ์ระหว่างสาย
# 9.  Correlation Pairs Ranking — Top คู่สายที่สัมพันธ์สูงสุด
# 10. Lag Correlation — พฤติกรรมการเปลี่ยนสาย
# 11. Ridership Market Share Over Time (Smoothed) — สัดส่วนตลาด
# 12. Ridership Distribution Box Plot
# 13. Calendar Heatmap (aggfunc=median)
# 14. Top 10 Ridership Days — วันผู้โดยสารสูงสุด
# 15. Summary — สรุปข้อค้นพบเชิงปริมาณ

# %% [markdown]
# ### 5.1 กราฟ Time Series — ผู้โดยสารรายวันแยกตามสาย (Multi-line Chart)

# %%
# แปลงเป็น long format สำหรับ multi-line time series
ts_long = (
    pivot_df[rail_lines]
    .reset_index()
    .melt(id_vars='date', var_name='สาย', value_name='ผู้โดยสาร')
)

fig = px.line(
    ts_long,
    x='date',
    y='ผู้โดยสาร',
    color='สาย',
    title='ปริมาณผู้โดยสารรายวันแยกตามสายรถไฟฟ้า (Daily Ridership by Rail Line)',
    labels={'date': 'วันที่', 'ผู้โดยสาร': 'ผู้โดยสาร (คน)', 'สาย': 'สายรถไฟฟ้า'},
)
fig.update_layout(hovermode='x unified')
fig.show()

# %% [markdown]
# ### 5.2 Normalized Ridership Index — เปรียบเทียบการเติบโตบน Scale เดียวกัน
#
# BTS ~800K/วัน vs ARL ~70K/วัน → เปรียบตรงๆ ไม่ได้
# แปลงเป็น Index (Base = 100) ณ จุดเริ่มต้น เพื่อดู Growth Dynamics ของแต่ละสาย

# %%
# ใช้ค่าเฉลี่ย 7 วันแรกที่ > 0 เป็นฐาน — ลด noise จากวันแรกที่อาจมีข้อมูลไม่ครบ
normalized_df = pivot_df[rail_lines].copy().astype(float)
for col in rail_lines:
    first_vals = normalized_df[col][normalized_df[col] > 0].head(7)
    base = first_vals.mean() if len(first_vals) > 0 else 1.0
    normalized_df[col] = normalized_df[col] / base * 100

fig = px.line(
    normalized_df.reset_index().melt(id_vars='date', var_name='สาย', value_name='ดัชนี'),
    x='date',
    y='ดัชนี',
    color='สาย',
    title='ดัชนีผู้โดยสาร (Ridership Index) — ฐาน = 100 ณ จุดเริ่มต้น',
    labels={'date': 'วันที่', 'ดัชนี': 'Index (Base = 100)', 'สาย': 'สายรถไฟฟ้า'},
)
fig.add_hline(y=100, line_dash='dash', line_color='gray')
# Annotation ชี้ Base Period ให้กรรมการเห็น
fig.add_annotation(
    x=normalized_df.index[6],
    y=100,
    text='Base Period (avg first 7 days)',
    showarrow=True,
    arrowhead=2,
    font=dict(size=11),
    bgcolor='lightyellow',
)
fig.update_layout(hovermode='x unified')
fig.show()

# %% [markdown]
# ### 5.3 จัดอันดับค่าเฉลี่ยผู้โดยสาร (Average Ridership Ranking)

# %%
# คำนวณค่าเฉลี่ย+std โดยแทน 0 ด้วย NaN (วันที่ไม่มีบริการไม่ควรลดค่าเฉลี่ย)
avg_s = pivot_df[rail_lines].replace(0, np.nan).mean()
std_s = pivot_df[rail_lines].replace(0, np.nan).std()

# ใช้ .fillna(0).astype(int) ป้องกัน NaN crash เมื่อแปลงเป็น int
rank_df = pd.DataFrame({
    'สาย':       avg_s.index,
    'avg_daily': avg_s.fillna(0).round().astype(int).values,
    'std_daily': std_s.fillna(0).round().astype(int).values,
})
rank_df['cv_pct'] = np.where(
    rank_df['avg_daily'] > 0,
    (rank_df['std_daily'] / rank_df['avg_daily'] * 100).round(1),
    0.0,
)
rank_df = rank_df.sort_values('avg_daily', ascending=False).reset_index(drop=True)

print('=== จัดอันดับค่าเฉลี่ยผู้โดยสารรายวัน ===')
print(rank_df.to_string(index=False))

fig = px.bar(
    rank_df,
    x='สาย',
    y='avg_daily',
    color='avg_daily',
    color_continuous_scale='Blues',
    title='ค่าเฉลี่ยผู้โดยสารรายวันแยกตามสาย (Avg Daily Ridership Ranking)',
    labels={'avg_daily': 'เฉลี่ยผู้โดยสาร/วัน', 'สาย': 'สายรถไฟฟ้า'},
    text='avg_daily',
)
fig.update_traces(texttemplate='%{text:,}', textposition='outside')
fig.show()

# %% [markdown]
# ### 5.4 ความผันผวน — Coefficient of Variation (CV)
#
# CV = std / mean × 100% — เปรียบเทียบความผันผวนข้ามสายที่มี scale ต่างกันอย่างยุติธรรม
# - CV สูง → อ่อนไหวต่อเทศกาล/เหตุการณ์พิเศษ
# - CV ต่ำ → ฐานผู้โดยสารประจำมั่นคง

# %%
fig = px.bar(
    rank_df.sort_values('cv_pct', ascending=False),
    x='สาย',
    y='cv_pct',
    color='cv_pct',
    color_continuous_scale='RdYlGn_r',
    title='ความผันผวนของผู้โดยสารแต่ละสาย (Coefficient of Variation %)',
    labels={'cv_pct': 'CV (%)', 'สาย': 'สายรถไฟฟ้า'},
    text='cv_pct',
)
fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
fig.show()

# %% [markdown]
# ### 5.5 Rolling 30-Day Std — ความผันผวนที่เปลี่ยนแปลงตามเวลา
#
# แทน 0 ด้วย NaN ก่อน rolling เพื่อไม่ให้วันที่ไม่มีบริการบิดเบือน Std

# %%
# replace(0, np.nan) ก่อน rolling ป้องกัน Std ถูกดึงต่ำโดยวันที่ไม่มีข้อมูล
rolling_std_df = (
    pivot_df[rail_lines]
    .replace(0, np.nan)
    .rolling(30)
    .std()
    .dropna()
)

fig = px.line(
    rolling_std_df.reset_index().melt(id_vars='date', var_name='สาย', value_name='Rolling Std'),
    x='date',
    y='Rolling Std',
    color='สาย',
    title='ความผันผวน Rolling 30 วันแต่ละสาย (Rolling 30-Day Standard Deviation)',
    labels={'date': 'วันที่', 'Rolling Std': 'Standard Deviation (30D)'},
)
fig.show()

# %% [markdown]
# ### 5.6 รูปแบบการเดินทางตามวันในสัปดาห์ (Weekday Seasonality)
#
# **Raw:** ค่าเฉลี่ยจริง → เห็นขนาดของแต่ละสาย
# **Normalized:** หาร mean ของแต่ละสาย → เห็น pattern จริงโดยไม่ถูกบิดเบือนโดย scale

# %%
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

weekday_avg = (
    pivot_df[rail_lines]
    .assign(day=pivot_df.index.day_name())
    .groupby('day')[rail_lines]
    .mean()
    .reindex(day_order)
    .round()
)

print('=== ค่าเฉลี่ยผู้โดยสารแยกตามวันในสัปดาห์ ===')
print(weekday_avg.to_string())

# Raw Grouped Bar Chart
fig = px.bar(
    weekday_avg.reset_index().melt(id_vars='day', var_name='สาย', value_name='avg_passengers'),
    x='day',
    y='avg_passengers',
    color='สาย',
    barmode='group',
    title='ค่าเฉลี่ยผู้โดยสารรายวันในสัปดาห์ — Raw (Avg Daily Ridership by Day of Week)',
    labels={'avg_passengers': 'เฉลี่ยผู้โดยสาร', 'day': 'วัน', 'สาย': 'สายรถไฟฟ้า'},
    category_orders={'day': day_order},
)
fig.show()

# %%
# Normalized: หาร mean ของแต่ละสายออก → ดู pattern โดยไม่ถูก scale ครอบงำ
# ค่า > 1.0 = วันนั้นสูงกว่าค่าเฉลี่ยของสายนั้น, < 1.0 = ต่ำกว่า
weekday_norm = weekday_avg.div(weekday_avg.mean())

fig = px.line(
    weekday_norm.reset_index().melt(id_vars='day', var_name='สาย', value_name='normalized'),
    x='day',
    y='normalized',
    color='สาย',
    markers=True,
    title='รูปแบบรายวัน Normalized — ค่า > 1 = สูงกว่าค่าเฉลี่ยของสายนั้น',
    labels={'normalized': 'Normalized Ridership (1.0 = avg)', 'day': 'วัน'},
    category_orders={'day': day_order},
)
fig.add_hline(y=1.0, line_dash='dash', line_color='gray', annotation_text='ค่าเฉลี่ย (1.0)')
fig.show()

# %% [markdown]
# ### 5.7 Rolling 30-Day YoY Growth — แนวโน้มการเติบโต

# %%
# หมายเหตุ: YoY นี้ใช้ total_passengers ซึ่งรวมทุกสาย
# หากมีการเปิดสายใหม่ใน 2026 อาจทำให้ YoY ดูสูงกว่าความเป็นจริง
pivot_df['rolling_30'] = pivot_df['total_passengers'].rolling(30).mean()
pivot_df['rolling_30_yoy'] = pivot_df['rolling_30'].pct_change(periods=365) * 100

yoy_plot = pivot_df.dropna(subset=['rolling_30_yoy'])

if len(yoy_plot) > 0:
    fig = px.line(
        yoy_plot.reset_index(),
        x='date',
        y='rolling_30_yoy',
        title='อัตราการเติบโต YoY แบบ Rolling 30 วัน (Rolling 30-Day YoY Growth %)',
        labels={'rolling_30_yoy': 'YoY Growth (%)', 'date': 'วันที่'},
    )
    fig.add_hline(y=0, line_dash='dash', line_color='red', annotation_text='เส้นฐาน 0%')
    fig.show()
else:
    print('⚠️ ข้อมูลไม่เพียงพอสำหรับ YoY Rolling')
    print(f'   ต้องการ ≥ 395 วัน — ปัจจุบัน {len(pivot_df)} วัน')

# %% [markdown]
# ### 5.8 Clustered Correlation Heatmap (Log-transform) — ความสัมพันธ์ระหว่างสาย
#
# ใช้ log1p transform ก่อน corr เพื่อจัดการ heteroskedasticity ของ ridership
# (ผู้โดยสารมักมี variance เพิ่มตาม level)
# ใช้ distance = 1 - corr สำหรับ Hierarchical Clustering (ถูกต้องกว่าใช้ corr โดยตรง)

# %%
import scipy.cluster.hierarchy as sch

# log1p transform ก่อน corr — สะท้อน relative movement ไม่ใช่ absolute
corr = np.log1p(pivot_df[rail_lines].replace(0, np.nan)).corr()

# Hierarchical Clustering ด้วย distance = 1 - corr
try:
    distance     = 1 - corr.fillna(0)
    linkage      = sch.linkage(distance.values, method='average')
    cluster_order = sch.leaves_list(linkage)
    corr_sorted  = corr.iloc[cluster_order, cluster_order]
except Exception:
    corr_sorted = corr

# zmin/zmax ต้องระบุชัดเจนเพื่อให้ scale ตรงกับความหมายของ correlation เสมอ
fig = ff.create_annotated_heatmap(
    z=np.round(corr_sorted.values, 2),
    x=corr_sorted.columns.tolist(),
    y=corr_sorted.index.tolist(),
    colorscale='RdBu',
    showscale=True,
    zmin=-1,
    zmax=1,
)
fig.update_layout(title='ความสัมพันธ์ผู้โดยสารระหว่างสาย — Clustered Log-transform (Ridership Correlation Heatmap)')
fig.show()

# %% [markdown]
# ### 5.9 Correlation Pairs Ranking — Top คู่สายที่มีความสัมพันธ์สูงสุด

# %%
# ดึงเฉพาะ upper triangle (หลีกเลี่ยงค่าซ้ำ) แล้วจัดอันดับ
corr_pairs = (
    corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    .stack()
    .sort_values(ascending=False)
    .reset_index()
)
corr_pairs.columns = ['สาย A', 'สาย B', 'Correlation']
corr_pairs['Correlation'] = corr_pairs['Correlation'].round(3)

print('=== Top 5 คู่สายที่มีความสัมพันธ์สูงสุด ===')
print(corr_pairs.head(5).to_string(index=False))
print('\n=== Bottom 5 คู่สายที่มีความสัมพันธ์ต่ำสุด ===')
print(corr_pairs.tail(5).to_string(index=False))

# %% [markdown]
# ### 5.10 Lag Correlation — พฤติกรรมการเปลี่ยนสาย (Transfer Behaviour)
#
# ทดสอบทั้งสองทิศทาง:
# - shift(-1): B วันถัดไป vs A วันนี้ → ทดสอบว่า A นำ B
# - shift(+1): B วันก่อน vs A วันนี้ → ทดสอบว่า B นำ A
# - threshold = 0.05 (แทน 0.02 เดิม) เพื่อลด false positive

# %%
lag_pairs_candidates = [
    ('BTS',               'MRT Blue'),
    ('BTS',               'MRT Purple'),
    ('Airport Rail Link', 'BTS'),
    ('MRT Blue',          'Airport Rail Link'),
]
lag_pairs = [(a, b) for a, b in lag_pairs_candidates if a in rail_lines and b in rail_lines]

LAG_THRESHOLD = 0.05   # ต้องต่างกันอย่างน้อย 0.05 จึงถือว่ามี directional effect

lag_results = []
for line_a, line_b in lag_pairs:
    s_a  = pivot_df[line_a]
    s_b  = pivot_df[line_b]
    lag0   = s_a.corr(s_b)
    lag_n1 = s_a.corr(s_b.shift(-1))   # A นำ B (B วันถัดไป)
    lag_p1 = s_a.corr(s_b.shift(1))    # B นำ A (B วันก่อน)

    if lag_n1 > lag0 + LAG_THRESHOLD:
        interpretation = f'{line_a} leads {line_b}'
    elif lag_p1 > lag0 + LAG_THRESHOLD:
        interpretation = f'{line_b} leads {line_a}'
    else:
        interpretation = 'Simultaneous demand'

    lag_results.append({
        'คู่สาย':       f'{line_a} ↔ {line_b}',
        'Lag-0':        round(lag0,   3),
        'Lag-1 (A→B)':  round(lag_n1, 3),
        'Lag-1 (B→A)':  round(lag_p1, 3),
        'สรุป':         interpretation,
    })

lag_df = pd.DataFrame(lag_results)
print(f'=== Lag Correlation Analysis (Transfer Behaviour, threshold={LAG_THRESHOLD}) ===')
print(lag_df.to_string(index=False))

# %% [markdown]
# ### 5.11 Ridership Market Share Over Time — สัดส่วนตลาดตามเวลา (Smoothed)
#
# ใช้ Rolling 14 วัน Smooth ก่อนแสดง เพื่อลด noise รายวันและเห็นแนวโน้มชัดขึ้น
# - สายที่สัดส่วนเพิ่ม → เติบโตเร็วกว่า Network โดยรวม
# - การเปลี่ยนแปลงกะทันหัน → อาจเกิดจากเปิดสายใหม่หรือปัญหาบริการ

# %%
# ป้องกัน divide-by-zero: แทน 0 ด้วย NaN ก่อนหาร
total_per_day = pivot_df[rail_lines].sum(axis=1).replace(0, np.nan)
share_time_df = (
    pivot_df[rail_lines]
    .div(total_per_day, axis=0)
    .multiply(100)
    .rolling(14)           # smooth 14 วัน — ลด noise ให้เห็นแนวโน้มชัด
    .mean()
    .dropna()
)

fig = px.area(
    share_time_df.reset_index().melt(id_vars='date', var_name='สาย', value_name='Share (%)'),
    x='date',
    y='Share (%)',
    color='สาย',
    title='สัดส่วนตลาดของแต่ละสายรถไฟฟ้าตามเวลา — Smoothed 14D (Rail Line Market Share Over Time %)',
    labels={'date': 'วันที่'},
)
fig.update_layout(yaxis_range=[0, 100])
fig.show()

# %% [markdown]
# ### 5.12 Ridership Distribution Box Plot — การกระจายตัวของผู้โดยสาร
#
# - Median → ค่ากลางผู้โดยสารรายวัน
# - IQR (Box width) → ความผันผวนปกติ
# - Outlier dots → วันพิเศษที่ผู้โดยสารพุ่งหรือร่วงผิดปกติ

# %%
box_df = pivot_df[rail_lines].replace(0, np.nan).melt(var_name='สาย', value_name='ผู้โดยสาร')

fig = px.box(
    box_df,
    x='สาย',
    y='ผู้โดยสาร',
    color='สาย',
    title='การกระจายตัวของผู้โดยสารรายวันแต่ละสาย (Daily Ridership Distribution)',
    labels={'ผู้โดยสาร': 'ผู้โดยสาร (คน)', 'สาย': 'สายรถไฟฟ้า'},
)
fig.show()

# %% [markdown]
# ### 5.13 Calendar Heatmap — ตารางปฏิทินผู้โดยสารรายวัน
#
# - แถบแนวนอนมืดวันเสาร์/อาทิตย์ → วันหยุดผู้โดยสารน้อย
# - คอลัมน์แนวตั้งเย็น → สัปดาห์วันหยุดยาว (สงกรานต์, ปีใหม่)
# - คอลัมน์สว่าง → สัปดาห์ผู้โดยสารพุ่งจากเหตุการณ์พิเศษ
#
# ใช้ aggfunc='median' (แทน mean) เพื่อความทนทานต่อ outlier รายวัน

# %%
cal_df = pivot_df[['total_passengers']].copy()
cal_df['week']      = cal_df.index.isocalendar().week.astype(int)
cal_df['weekday']   = cal_df.index.weekday    # 0 = Monday
cal_df['year_week'] = (
    cal_df.index.year.astype(str) + '-W' +
    cal_df['week'].astype(str).str.zfill(2)
)

# aggfunc='median' — ทนทานต่อ outlier มากกว่า mean
pivot_cal = cal_df.pivot_table(
    index='weekday',
    columns='year_week',
    values='total_passengers',
    aggfunc='median',
)

# reindex(range(7)) ให้แน่ใจว่าลำดับวัน 0–6 ถูกต้องก่อน map ป้ายกำกับ
pivot_cal = pivot_cal.reindex(range(7))
day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

fig = go.Figure(go.Heatmap(
    z=pivot_cal.values,
    x=pivot_cal.columns,
    y=day_labels,
    colorscale='YlOrRd',
    colorbar=dict(title='Median ผู้โดยสาร'),
))
fig.update_layout(
    title='ปฏิทินผู้โดยสารรายวัน — ทุกสายรถไฟฟ้า (Daily Ridership Calendar Heatmap)',
    xaxis_title='สัปดาห์',
    yaxis_title='วันในสัปดาห์',
)
fig.show()

# %% [markdown]
# ### 5.14 Top 10 Ridership Days — วันที่ผู้โดยสารสูงสุด (พร้อม Day-of-Week Annotation)

# %%
# ค้นหา 10 วันที่มีผู้โดยสารรวมสูงสุด — มักสัมพันธ์กับเหตุการณ์พิเศษ
top_days = (
    pivot_df[['total_passengers']]
    .nlargest(10, 'total_passengers')
    .reset_index()
)
top_days['day_of_week'] = top_days['date'].dt.day_name()

print('=== Top 10 วันที่มีผู้โดยสารสูงสุด ===')
print(top_days.to_string(index=False))

fig = px.bar(
    top_days,
    x='date',
    y='total_passengers',
    color='day_of_week',
    title='Top 10 วันที่มีผู้โดยสารรวมสูงสุด (จากทุกสาย)',
    labels={'total_passengers': 'ผู้โดยสารรวม (คน)', 'date': 'วันที่', 'day_of_week': 'วัน'},
    text='total_passengers',
)
fig.update_traces(texttemplate='%{text:,}', textposition='outside')

# Annotate ชื่อย่อวัน (Mon/Tue/...) บน bar เพื่อให้กรรมการเห็น pattern ทันที
for _, row in top_days.iterrows():
    fig.add_annotation(
        x=row['date'],
        y=row['total_passengers'],
        text=row['day_of_week'][:3],
        showarrow=False,
        yshift=18,
        font=dict(size=10, color='navy'),
    )
fig.show()

# %% [markdown]
# ### 5.15 สรุปข้อค้นพบ Phase 5 (Urban Rail Comparison Insights)

# %%
# สรุปเชิงปริมาณ — ส่งต่อใน Phase 9 (Insights & Conclusion)
top_line       = rank_df.iloc[0]
most_volatile  = rank_df.sort_values('cv_pct', ascending=False).iloc[0]
least_volatile = rank_df.sort_values('cv_pct').iloc[0]

# อัตราส่วน Weekday vs Weekend (เฉลี่ยทุกสาย)
weekday_mean = weekday_avg.loc[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']].mean().mean()
weekend_mean = weekday_avg.loc[['Saturday', 'Sunday']].mean().mean()
wd_we_ratio  = weekday_mean / weekend_mean if weekend_mean > 0 else float('nan')

top_corr_pair = corr_pairs.iloc[0]

print('=== Urban Rail Comparison Summary ===')
print(f'สายผู้โดยสารเฉลี่ยสูงสุด:        {top_line["สาย"]} ({int(top_line["avg_daily"]):,} คน/วัน)')
print(f'สายความผันผวนสูงสุด (CV):        {most_volatile["สาย"]} ({most_volatile["cv_pct"]:.1f}%)')
print(f'สายความผันผวนต่ำสุด (CV):        {least_volatile["สาย"]} ({least_volatile["cv_pct"]:.1f}%)')
print(f'อัตราส่วน Weekday/Weekend:       {wd_we_ratio:.2f}×  (weekday {weekday_mean:,.0f} vs weekend {weekend_mean:,.0f} คน/วัน)')
print(f'คู่สายที่มีความสัมพันธ์สูงสุด:   {top_corr_pair["สาย A"]} ↔ {top_corr_pair["สาย B"]} (r = {top_corr_pair["Correlation"]:.3f})')
