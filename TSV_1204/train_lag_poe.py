import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

# ==========================================
# 1. 模拟生成数据 (替换为你真实的 Excel 读取代码)
# ==========================================
def generate_mock_data():
    """
    模拟生成 3片 Wafer * 14 Groups * 17 Points 的数据
    结构: [Wafer_ID, Group_ID, TSV_ID, CD(线宽), Density(密度), Depth(刻蚀深度)]
    """
    np.random.seed(42)
    data = []
    
    for wafer in [1, 2, 3]:
        for group in range(1, 15): # 14 groups
            # 模拟该 Group 的基础刻蚀速率 (受机台状态影响)
            base_rate = 100 + np.random.normal(0, 2) 
            
            for point in range(1, 18): # 17 points
                # 模拟物理特征
                # CD: 假设有几种不同规格的孔径，范围 2um - 10um
                cd = np.random.choice([2.0, 2.5, 5.0, 8.0, 10.0]) 
                # Density: 局部密度
                density = np.random.uniform(0.1, 0.5)
                
                # 模拟真实的 Lag 效应物理公式 (仅用于生成数据，模型不知道这个公式)
                # 规律：CD越小，Depth越小 (Lag效应); Density越大，Depth越小 (Loading效应)
                lag_factor = 1 - (1 / (cd + 0.5)) * 0.5 
                loading_factor = 1 - density * 0.2
                
                depth = base_rate * lag_factor * loading_factor + np.random.normal(0, 0.5)
                
                data.append({
                    'Wafer_ID': wafer,
                    'Group_ID': f'W{wafer}_G{group}', # 唯一标识一个Group
                    'TSV_ID': point,
                    'CD': cd,
                    'Density': density,
                    'Depth': depth
                })
    
    return pd.DataFrame(data)

# ==========================================
# 2. 数据配对与特征工程 (核心步骤)
# ==========================================
def create_pairs(df):
    """
    在每个 Group 内部进行两两配对。
    目标：预测 (Depth_B - Depth_A)
    """
    pair_data = []
    
    # 按 Group 分组处理
    groups = df.groupby('Group_ID')
    
    for group_name, group_df in groups:
        # 获取该组内所有样本的索引
        indices = group_df.index.tolist()
        
        # 生成排列组合 (A, B)，其中 A 是参考点，B 是目标点
        # 我们可以限制策略：只生成 CD_A < CD_B 的配对，或者全配对
        # 这里采用全配对，让模型学习正负Lag
        for idx_a, idx_b in itertools.permutations(indices, 2):
            row_a = group_df.loc[idx_a]
            row_b = group_df.loc[idx_b]
            
            # 构建特征
            pair_sample = {
                'Wafer_ID': row_a['Wafer_ID'],
                'Group_ID': group_name,
                
                # 原始特征 A (参考点)
                'CD_Ref': row_a['CD'],
                'Density_Ref': row_a['Density'],
                
                # 原始特征 B (目标点)
                'CD_Target': row_b['CD'],
                'Density_Target': row_b['Density'],
                
                # *** 关键交互特征 ***
                # 模型最需要知道的是“差异”
                'Delta_CD': row_b['CD'] - row_a['CD'],
                'Ratio_CD': row_b['CD'] / (row_a['CD'] + 1e-6),
                'Delta_Density': row_b['Density'] - row_a['Density'],
                
                # 目标变量 (Label): 深度差
                'Delta_Depth': row_b['Depth'] - row_a['Depth']
            }
            pair_data.append(pair_sample)
            
    return pd.DataFrame(pair_data)

# ==========================================
# 3. 模型训练与验证
# ==========================================
def train_lgbm_gpu(df_pairs):
    # 定义特征列
    features = [
        'CD_Ref', 'Density_Ref', 
        'CD_Target', 'Density_Target', 
        'Delta_CD', 'Ratio_CD', 'Delta_Density'
    ]
    target = 'Delta_Depth'
    
    X = df_pairs[features]
    y = df_pairs[target]
    groups = df_pairs['Wafer_ID'] # 用于分组交叉验证
    
    # LightGBM 参数 (针对 GPU 优化)
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'device': 'gpu',           # 启用 GPU
        'gpu_platform_id': 0,      # 第一个计算平台
        'gpu_device_id': 0,        # 使用第一张 4090
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': -1,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1
    }
    
    # 使用 GroupKFold 进行交叉验证
    # 这非常重要！确保同一片 Wafer 的数据要么都在训练集，要么都在测试集
    # 防止模型记住某片 Wafer 的特定偏差
    gkf = GroupKFold(n_splits=3) 
    
    oof_preds = np.zeros(len(X))
    scores = []
    
    print(f"开始训练... 数据总量: {len(X)} 行")
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        # 构建数据集
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
        
        # 训练
        callbacks = [
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
        
        model = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_train, lgb_val],
            callbacks=callbacks
        )
        
        # 预测
        val_preds = model.predict(X_val)
        oof_preds[val_idx] = val_preds
        
        # 评估
        rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        scores.append(rmse)
        print(f"Fold {fold+1} RMSE: {rmse:.4f}")
        
        # 保存最后一个模型用于分析特征重要性
        final_model = model

    print(f"\n平均 RMSE: {np.mean(scores):.4f}")
    print(f"总体 R2 Score: {r2_score(y, oof_preds):.4f}")
    
    return final_model, features, y, oof_preds

# ==========================================
# 4. 结果可视化
# ==========================================
def plot_results(model, features, y_true, y_pred):
    plt.figure(figsize=(12, 5))
    
    # 1. 预测值 vs 真实值
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.1)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('Actual Delta Depth')
    plt.ylabel('Predicted Delta Depth')
    plt.title('Prediction Accuracy')
    
    # 2. 特征重要性
    plt.subplot(1, 2, 2)
    importance = model.feature_importance(importance_type='gain')
    sns.barplot(x=importance, y=features)
    plt.title('Feature Importance (Gain)')
    
    plt.tight_layout()
    plt.show()

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    # 1. 获取数据 (此处用模拟数据，请替换为 pd.read_excel('your_file.xlsx'))
    print("正在生成/读取数据...")
    df_raw = generate_mock_data()
    
    # 2. 配对处理
    print("正在进行数据配对...")
    df_pairs = create_pairs(df_raw)
    print(f"配对后数据量: {len(df_pairs)}")
    
    # 3. 训练
    model, feats, y_true, y_pred = train_lgbm_gpu(df_pairs)
    
    # 4. 展示结果
    plot_results(model, feats, y_true, y_pred)
    
    print("\n=== 预测应用示例 ===")
    print("假设已知一个参考孔深度为 100um (CD=2um), 预测旁边一个 CD=8um 的孔深度:")
    # 构造输入向量
    test_input = pd.DataFrame([{
        'CD_Ref': 2.0, 'Density_Ref': 0.3,
        'CD_Target': 8.0, 'Density_Target': 0.3,
        'Delta_CD': 6.0, 'Ratio_CD': 4.0, 'Delta_Density': 0.0
    }])
    pred_delta = model.predict(test_input)[0]
    print(f"预测深度差 (Delta): {pred_delta:.2f} um")
    print(f"预测目标孔深度: {100 + pred_delta:.2f} um")