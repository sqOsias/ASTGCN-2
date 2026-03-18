import os
import json
import pandas as pd

def summarize_experiments(base_dir="results", output_csv="experiment_summary.csv"):
    summary_data =[]

    # 遍历所有的 results 目录
    for root, dirs, files in os.walk(base_dir):
        # 通过寻找 resolved_config.json 来精确定位一个完整的实验配置目录
        if 'resolved_config.json' in files and os.path.basename(root) == 'configs':
            # 获取当前实验的主目录 (例如: results/ASTGCN_lr0p001/0_0_20260318033244)
            run_dir = os.path.dirname(root)
            run_id = os.path.basename(run_dir)             # 如: 0_0_20260318033244
            run_group = os.path.basename(os.path.dirname(run_dir)) # 如: ASTGCN_lr0p001
            
            # 初始化当前实验的记录字典
            row = {
                'Run_Group': run_group,
                'Run_ID': run_id,
                'Spatial_Mode': -1,
                'Temporal_Mode': -1,
                'Best_Epoch': -1,
                'Val_Loss': None
            }

            # ==========================================
            # 1. 解析配置参数 (resolved_config.json)
            # ==========================================
            config_path = os.path.join(root, 'resolved_config.json')
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    cfg = json.load(f)
                    row['Model'] = cfg.get('Training', {}).get('model_name', 'ASTGCN')
                    row['Learning_Rate'] = float(cfg.get('Training', {}).get('learning_rate', 0))
                    row['Batch_Size'] = int(cfg.get('Training', {}).get('batch_size', 0))
                    
                    upgrade_cfg = cfg.get('ModelUpgrade', {})
                    row['Spatial_Mode'] = upgrade_cfg.get('spatial_mode', 0)
                    row['Temporal_Mode'] = upgrade_cfg.get('temporal_mode', 0)
            except Exception as e:
                print(f"[-] 读取配置失败 {run_id}: {e}")

            # ==========================================
            # 2. 解析运行时硬件指标 (artifacts/runtime.json)
            # ==========================================
            runtime_path = os.path.join(run_dir, 'artifacts', 'runtime.json')
            if os.path.exists(runtime_path):
                try:
                    with open(runtime_path, 'r', encoding='utf-8') as f:
                        rt = json.load(f)
                        row['Train_Time(s)'] = round(rt.get('train_seconds', 0), 2)
                        # 将峰值显存从 Byte 转换为 MB
                        row['Peak_Mem(MB)'] = round(rt.get('gpu_peak_bytes', 0) / (1024 * 1024), 2)
                except:
                    pass

            # ==========================================
            # 3. 解析模型指标 (metrics 目录)
            # ==========================================
            val_csv = os.path.join(run_dir, 'metrics', 'val_metrics.csv')
            test_csv = os.path.join(run_dir, 'metrics', 'test_metrics.csv')

            if os.path.exists(val_csv) and os.path.exists(test_csv):
                try:
                    val_df = pd.read_csv(val_csv)
                    # 排除第 0 个 Epoch（初始化的结果），寻找真正训练后验证集 Loss 最低的 Epoch
                    if len(val_df) > 1:
                        val_df = val_df[val_df['epoch'] > 0]
                    
                    best_idx = val_df['validation_loss'].idxmin()
                    best_epoch = int(val_df.loc[best_idx, 'epoch'])
                    row['Best_Epoch'] = best_epoch
                    row['Val_Loss'] = round(val_df.loc[best_idx, 'validation_loss'], 4)

                    # 读取该 Best Epoch 对应的测试集指标 (MAE, RMSE, MAPE)
                    test_df = pd.read_csv(test_csv)
                    best_test = test_df[test_df['epoch'] == best_epoch]
                    
                    # 动态写入不同预测步长 (horizon) 的结果
                    for _, t_row in best_test.iterrows():
                        hz = int(t_row['horizon'])
                        row[f'MAE_step{hz}'] = round(t_row['MAE'], 4)
                        row[f'RMSE_step{hz}'] = round(t_row['RMSE'], 4)
                        row[f'MAPE_step{hz}'] = round(t_row['MAPE'], 4)
                        
                except Exception as e:
                    print(f"[-] 解析评估指标失败 {run_id}: {e}")

            summary_data.append(row)

    # 汇总并保存为 DataFrame
    if not summary_data:
        print("[!] 没有在指定目录找到任何完整的实验结果。请检查路径。")
        return

    df = pd.DataFrame(summary_data)
    
    # 按照 Spatial_Mode 和 Temporal_Mode 排序，方便你看消融实验对比
    df.sort_values(by=['Model', 'Spatial_Mode', 'Temporal_Mode', 'Run_ID'], inplace=True)
    
    # 保存为 CSV
    df.to_csv(output_csv, index=False)
    print(f"\n[+] 成功汇总 {len(df)} 组实验结果！已保存至 -> {output_csv}")

if __name__ == "__main__":
    # 确保基础路径是你实际保存的 results 或 params 目录
    # 注意：根据你的截图，目录叫 'results'
    summarize_experiments(base_dir="results", output_csv="experiment_summary.csv")