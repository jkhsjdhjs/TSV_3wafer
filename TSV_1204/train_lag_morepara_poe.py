import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. 数据加载与智能清洗
# ==========================================
print("Loading data...")
# 读取所有数据，不设 header
df_raw = pd.read_excel('tsv_5p_rawdata_sum.xlsx', header=None)

# 1.1 识别 Header 行 (包含 'up_width')
header_rows = df_raw[df_raw[0].astype(str).str.contains('up_width', na=False)].index.tolist()
print(f"Detected {len(header_rows)} header rows (data blocks)")

if len(header_rows) == 0:
    raise ValueError("No header found (looking for 'up_width').")

# 取第一个 header 行作为列名
cols = df_raw.iloc[header_rows[0]].values

# 1.2 构建数据：每个 header 后的数据块是一个 Site 的数据
# 每 14 个数据块 = 1 个 Wafer
sites_per_wafer = 14
data_list = []

for block_idx, header_idx in enumerate(header_rows):
    # 确定当前数据块的范围
    start_idx = header_idx + 1
    end_idx = header_rows[block_idx + 1] if block_idx + 1 < len(header_rows) else len(df_raw)
    
    # 提取数据块（通常只有1行或几行，取第一行有效数据）
    block_data = df_raw.iloc[start_idx:end_idx]
    
    # 只取第一行（假设每个 Site 只有一行测量数据）
    if len(block_data) > 0:
        row_data = block_data.iloc[0].values
        
        # 计算 Wafer_ID 和 Site_ID
        wafer_id = (block_idx // sites_per_wafer) + 1
        site_id = (block_idx % sites_per_wafer) + 1
        
        data_list.append(list(row_data) + [wafer_id, site_id])

# 构建 DataFrame
df_clean = pd.DataFrame(data_list, columns=list(cols) + ['Wafer_ID', 'Site_ID'])
print(f"Shape after initial cleaning: {df_clean.shape}")

# 1.3 类型转换与重命名
df_clean = df_clean.rename(columns={
    df_clean.columns[0]: 'CD',        # Width
    df_clean.columns[1]: 'Bottom', 
    df_clean.columns[2]: 'Depth', 
    df_clean.columns[3]: 'SD',        # Scallop Depth (scallps_length)
    df_clean.columns[4]: 'SW',        # Scallop Width (scallps_height)
    df_clean.columns[5]: 'Angle'      # Angle from file
})

numeric_cols = ['CD', 'Bottom', 'Depth', 'SW', 'SD', 'Angle', 'Wafer_ID', 'Site_ID']
for col in numeric_cols:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

df_clean = df_clean.dropna(subset=['CD', 'Depth'])

print(f"Total valid rows: {len(df_clean)}")
print(f"Number of Wafers detected: {df_clean['Wafer_ID'].nunique()}")
print(f"Rows per Wafer:\n{df_clean.groupby('Wafer_ID').size()}")

if len(df_clean) == 0:
    raise ValueError("Cleaned dataset is empty. Check data loading and column data types.")

# ==========================================
# 2. 数据配对 (Within-Wafer, Enhanced)
# ==========================================
print("\n=== Pairing Strategy ===")
print("Generating all possible pairs within each wafer")
print("Filtering out pairs with small CD difference (< 0.5 um)")

def create_pairs_within_wafer_enhanced(df, cd_threshold=0.5):
    """
    Enhanced pairing with more data:
    - All within-wafer pairs
    - Filter only pairs with |CD_diff| >= cd_threshold
    """
    pairs = []
    
    # 按 Wafer_ID 分组，只在组内配对
    for w_id, group in df.groupby('Wafer_ID'):
        records = group.to_dict('records')
        n = len(records)
        
        for i in range(n):
            for j in range(n):
                if i == j: continue
                
                t_row = records[i] # Target TSV (TSV2)
                r_row = records[j] # Reference TSV (TSV1)
                
                # 核心过滤：CD 差异必须大于阈值
                cd_diff = abs(t_row['CD'] - r_row['CD'])
                if cd_diff < cd_threshold:
                    continue
                    
                pair = {
                    'Wafer_ID': w_id,
                    
                    # --- Input Features (X) ---
                    # TSV1 (Reference) 的全部参数
                    'R_CD': r_row['CD'],           # TSV1 width
                    'R_Depth': r_row['Depth'],     # TSV1 depth
                    'R_Angle': r_row['Angle'],     # TSV1 angle
                    'R_SW': r_row['SW'],           # TSV1 scallop_height
                    'R_SD': r_row['SD'],           # TSV1 scallop_length
                    
                    # TSV2 的 width (已知，作为控制输入)
                    'T_CD': t_row['CD'],           # TSV2 width (INPUT)
                    
                    # --- Target Variables (Y) ---
                    # TSV2 的其余参数（待预测）
                    'T_Depth': t_row['Depth'],     # TSV2 depth (OUTPUT)
                    'T_Angle': t_row['Angle'],     # TSV2 angle (OUTPUT)
                    'T_SW': t_row['SW'],           # TSV2 scallop_height (OUTPUT)
                    'T_SD': t_row['SD']            # TSV2 scallop_length (OUTPUT)
                }
                pairs.append(pair)
            
    return pd.DataFrame(pairs)

df_pairs = create_pairs_within_wafer_enhanced(df_clean, cd_threshold=0.5)
print(f"Generated {len(df_pairs)} pairs (Within-Wafer, CD diff >= 0.5 um)")
print(f"Pairs per Wafer: {df_pairs.groupby('Wafer_ID').size().to_dict()}")

# ==========================================
# 3. 特征工程
# ==========================================
# 增加基础交互特征
df_pairs['R_AR'] = df_pairs['R_Depth'] / (df_pairs['R_CD'] + 1e-5)
df_pairs['CD_Ratio'] = df_pairs['T_CD'] / (df_pairs['R_CD'] + 1e-5)
df_pairs['CD_Delta'] = df_pairs['T_CD'] - df_pairs['R_CD']

# --- Enhanced Lag Features ---
df_pairs['T_CD_Sq'] = df_pairs['T_CD'] ** 2
df_pairs['T_CD_Inv'] = 1 / (df_pairs['T_CD'] + 1e-5)
df_pairs['Lag_Factor'] = df_pairs['R_Depth'] * df_pairs['CD_Ratio']
df_pairs['AR_Interaction'] = df_pairs['R_AR'] * df_pairs['CD_Ratio']

# ==========================================
# 4. 数据集划分 (70:15:15, Pair-Level Random Split)
# ==========================================
print("\n=== Data Split (Pair-Level) ===")
print("Random split at pair level (each pair is an independent prediction task)")
print("Note: Within-wafer pairing ensures no direct measurement leakage")

from sklearn.model_selection import train_test_split

# Step 1: Split into (Train+Val) and Test (85% / 15%)
train_val_df, test_df = train_test_split(df_pairs, test_size=0.15, random_state=42)

# Step 2: Split (Train+Val) into Train and Val
# 0.15 / 0.85 ≈ 0.1765
train_df, val_df = train_test_split(train_val_df, test_size=0.1765, random_state=42)

print(f"Train: {len(train_df)} pairs ({len(train_df)/len(df_pairs):.1%}), Wafers: {sorted(train_df['Wafer_ID'].unique())}")
print(f"Val:   {len(val_df)} pairs ({len(val_df)/len(df_pairs):.1%}), Wafers: {sorted(val_df['Wafer_ID'].unique())}")
print(f"Test:  {len(test_df)} pairs ({len(test_df)/len(df_pairs):.1%}), Wafers: {sorted(test_df['Wafer_ID'].unique())}")

# ==========================================
# 5. 链式建模
# ==========================================
# 核心输入特征：仅使用 TSV1 的参数 + TSV2 的 CD
core_input_feats = ['R_CD', 'R_Depth', 'R_Angle', 'R_SW', 'R_SD',  # TSV1 全部
                    'T_CD']  # TSV2 的 width

# 增强特征（基于核心输入计算得到）
enhanced_feats = ['R_AR', 'CD_Ratio', 'CD_Delta', 'T_CD_Sq', 'T_CD_Inv', 
                  'Lag_Factor', 'AR_Interaction']

base_feats = core_input_feats + enhanced_feats

lgb_params = {
    'n_estimators': 2000, 
    'learning_rate': 0.03, 
    'num_leaves': 31,
    'n_jobs': -1, 
    'random_state': 42, 
    'verbose': -1
}

def evaluate_metrics(y_true, y_pred, name="Model"):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    
    print(f"\n--- {name} Metrics ---")
    print(f"R2:   {r2:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.2f}%")
    return r2, mae, rmse

# --- Model 1: Predict T_Depth ---
print("\n" + "="*50)
print("[Model 1] Training Depth Prediction")
print("="*50)
print(f"Input: {core_input_feats}")
print(f"Output: T_Depth")

y_train_depth = np.log(train_df['T_Depth']) - np.log(train_df['R_Depth'])
y_val_depth = np.log(val_df['T_Depth']) - np.log(val_df['R_Depth'])

model_depth = lgb.LGBMRegressor(**lgb_params)
model_depth.fit(
    train_df[base_feats], y_train_depth,
    eval_set=[(val_df[base_feats], y_val_depth)],
    eval_metric='l2',
    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=0)]
)

def predict_depth(df, model):
    pred_log_ratio = model.predict(df[base_feats])
    return df['R_Depth'] * np.exp(pred_log_ratio)

train_df['Pred_T_Depth'] = predict_depth(train_df, model_depth)
val_df['Pred_T_Depth'] = predict_depth(val_df, model_depth)
test_df['Pred_T_Depth'] = predict_depth(test_df, model_depth)

evaluate_metrics(test_df['T_Depth'], test_df['Pred_T_Depth'], "Test Depth")

# --- Model 2: Predict T_Angle (with Tan Transformation) ---
print("\n" + "="*50)
print("[Model 2] Training Angle Prediction (Tan-based)")
print("="*50)
print("Strategy: Predict tan(90° - Angle) for better linearity")

# 添加预测的深宽比特征
for df in [train_df, val_df, test_df]:
    df['Pred_T_AR'] = df['Pred_T_Depth'] / (df['T_CD'] + 1e-5)

feats_angle = base_feats + ['Pred_T_Depth', 'Pred_T_AR']
print(f"Input: {core_input_feats} + Pred_T_Depth")
print(f"Output: tan(90° - T_Angle)")

# 转换 Angle 到 tan 空间
# tan(90° - angle) = cot(angle) = 1/tan(angle)
# 对于接近 90° 的角度，使用 tan(90° - angle) 更线性
def angle_to_tan(angle):
    """Convert angle to tan(90° - angle)"""
    return np.tan(np.radians(90 - angle))

def tan_to_angle(tan_val):
    """Convert tan(90° - angle) back to angle"""
    return 90 - np.degrees(np.arctan(tan_val))

# 在 tan 空间中预测差值
train_df['R_Angle_Tan'] = angle_to_tan(train_df['R_Angle'])
train_df['T_Angle_Tan'] = angle_to_tan(train_df['T_Angle'])
val_df['R_Angle_Tan'] = angle_to_tan(val_df['R_Angle'])
val_df['T_Angle_Tan'] = angle_to_tan(val_df['T_Angle'])
test_df['R_Angle_Tan'] = angle_to_tan(test_df['R_Angle'])
test_df['T_Angle_Tan'] = angle_to_tan(test_df['T_Angle'])

y_train_angle_tan = train_df['T_Angle_Tan'] - train_df['R_Angle_Tan']
y_val_angle_tan = val_df['T_Angle_Tan'] - val_df['R_Angle_Tan']

model_angle = lgb.LGBMRegressor(**lgb_params)
model_angle.fit(
    train_df[feats_angle], y_train_angle_tan,
    eval_set=[(val_df[feats_angle], y_val_angle_tan)],
    eval_metric='l2',
    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=0)]
)

# 预测并转换回角度空间
train_df['Pred_T_Angle_Tan'] = train_df['R_Angle_Tan'] + model_angle.predict(train_df[feats_angle])
val_df['Pred_T_Angle_Tan'] = val_df['R_Angle_Tan'] + model_angle.predict(val_df[feats_angle])
test_df['Pred_T_Angle_Tan'] = test_df['R_Angle_Tan'] + model_angle.predict(test_df[feats_angle])

train_df['Pred_T_Angle'] = tan_to_angle(train_df['Pred_T_Angle_Tan'])
val_df['Pred_T_Angle'] = tan_to_angle(val_df['Pred_T_Angle_Tan'])
test_df['Pred_T_Angle'] = tan_to_angle(test_df['Pred_T_Angle_Tan'])

evaluate_metrics(test_df['T_Angle'], test_df['Pred_T_Angle'], "Test Angle")

# --- Model 3: Predict T_SW (Scallop Height) - Hybrid Approach ---
print("\n" + "="*50)
print("[Model 3] Training Scallop Height (SW) Prediction")
print("="*50)
print("Strategy: Chain model + TSV2's complete geometry (CD, Depth, Angle)")

# 添加 TSV2 的深宽比
for df in [train_df, val_df, test_df]:
    df['T_AR'] = df['Pred_T_Depth'] / (df['T_CD'] + 1e-5)

# 混合特征集：原始链式特征 + 明确的 TSV2 几何
feats_sw = feats_angle + ['Pred_T_Angle', 'T_AR']

print(f"Input: Base features + Pred_Depth + Pred_Angle + TSV2_AR")
print(f"Key insight: TSV2 geometry (T_CD, Pred_T_Depth, Pred_T_Angle) now explicit")
print(f"Output: T_SW (scallop_height)")

y_train_sw = train_df['T_SW'] - train_df['R_SW']
y_val_sw = val_df['T_SW'] - val_df['R_SW']

model_sw = lgb.LGBMRegressor(**lgb_params)
model_sw.fit(
    train_df[feats_sw], y_train_sw,
    eval_set=[(val_df[feats_sw], y_val_sw)],
    eval_metric='l2',
    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=0)]
)

test_df['Pred_T_SW'] = test_df['R_SW'] + model_sw.predict(test_df[feats_sw])
evaluate_metrics(test_df['T_SW'], test_df['Pred_T_SW'], "Test SW")

# --- Model 4: Predict T_SD (Scallop Length) - Hybrid Approach ---
print("\n" + "="*50)
print("[Model 4] Training Scallop Length (SD) Prediction")
print("="*50)
print("Strategy: Chain model + TSV2's complete geometry (CD, Depth, Angle)")

feats_sd = feats_sw  # Same as SW
print(f"Input: Base features + Pred_Depth + Pred_Angle + TSV2_AR")
print(f"Key insight: TSV2 geometry (T_CD, Pred_T_Depth, Pred_T_Angle) now explicit")
print(f"Output: T_SD (scallop_length)")

y_train_sd = train_df['T_SD'] - train_df['R_SD']
y_val_sd = val_df['T_SD'] - val_df['R_SD']

model_sd = lgb.LGBMRegressor(**lgb_params)
model_sd.fit(
    train_df[feats_sd], y_train_sd,
    eval_set=[(val_df[feats_sd], y_val_sd)],
    eval_metric='l2',
    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=0)]
)

test_df['Pred_T_SD'] = test_df['R_SD'] + model_sd.predict(test_df[feats_sd])
evaluate_metrics(test_df['T_SD'], test_df['Pred_T_SD'], "Test SD")

# ==========================================
# 6. Visualization
# ==========================================
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.flatten()

metrics_plot = [
    ('T_Depth', 'Pred_T_Depth', 'Depth (um)'),
    ('T_Angle', 'Pred_T_Angle', 'Angle (degree)'),
    ('T_SW', 'Pred_T_SW', 'Scallop Height (um)'),
    ('T_SD', 'Pred_T_SD', 'Scallop Length (um)')
]

for i, (true_col, pred_col, title) in enumerate(metrics_plot):
    r2 = r2_score(test_df[true_col], test_df[pred_col])
    rmse = np.sqrt(mean_squared_error(test_df[true_col], test_df[pred_col]))
    mae = mean_absolute_error(test_df[true_col], test_df[pred_col])
    
    # 绘制散点
    axes[i].scatter(test_df[true_col], test_df[pred_col], alpha=0.5, s=50, color='blue', edgecolors='black', linewidths=0.5)
    
    # 绘制参考线
    min_val = min(test_df[true_col].min(), test_df[pred_col].min())
    max_val = max(test_df[true_col].max(), test_df[pred_col].max())
    axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    axes[i].set_title(f'{title}\nR²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}', fontsize=12, fontweight='bold')
    axes[i].set_xlabel('True Value', fontsize=11)
    axes[i].set_ylabel('Predicted Value', fontsize=11)
    axes[i].legend()
    axes[i].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('prediction_results.png', dpi=300, bbox_inches='tight')
print("\nPlot saved to: prediction_results.png")
plt.show()
