import matplotlib.pyplot as plt
import numpy as np

# 论文全局格式设置
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 600

# 数据准备 (请替换为您图片中的真实数据)
models = ['LSTM', 'STGCN', 'ASTGCN', 'AST-Informer(Ours)']
mae_scores = [4.21, 3.85, 3.42, 2.98] 
rmse_scores = [6.54, 5.92, 5.15, 4.30]

x = np.arange(len(models))
width = 0.35  # 柱子宽度

fig, ax = plt.subplots(figsize=(8, 5))
rects1 = ax.bar(x - width/2, mae_scores, width, label='MAE', color='#4A90E2', edgecolor='black', zorder=3)
rects2 = ax.bar(x + width/2, rmse_scores, width, label='RMSE', color='#F5A623', edgecolor='black', zorder=3)

# 图表装饰
ax.set_ylabel('Error Value')
ax.set_title('Prediction Performance Comparison at 60 Mins')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

# 增加水平网格线，使图表看起来更专业
ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

# 自动在柱子上方标注数值
ax.bar_label(rects1, padding=3, fmt='%.2f')
ax.bar_label(rects2, padding=3, fmt='%.2f')

fig.tight_layout()

# 导出为高精度矢量图
plt.savefig('performance_comparison.pdf', format='pdf', bbox_inches='tight')
plt.show()