#!/bin/bash
# 启动实时车辆速度预测系统

echo "正在启动实时车辆速度预测系统..."

# 检查并终止已存在的进程
# 终止已存在的进程
pids=$(lsof -i :8003 -t 2>/dev/null)
if [ ! -z "$pids" ]; then
    echo "终止已存在的进程: $pids"
    kill -9 $pids 2>/dev/null
fi

# 启动后端服务
cd /root/ASTGCN-2
echo "启动后端服务..."
uvicorn backend.__main__:app --host 0.0.0.0 --port 8003 &

# 等待服务启动
sleep 3

echo "系统已启动！"
echo "请访问 http://localhost:8003 查看数据大屏"
echo "按 Ctrl+C 停止服务"