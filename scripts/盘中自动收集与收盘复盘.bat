@echo off
chcp 65001 >nul
REM ============ 盘中五档买手/卖手自动收集与收盘准确率复盘 ============
REM 用法 1（双击直接运行，或任务计划每天 09:28 启动盘中自动采集+手机看盘）：
REM   盘中自动收集与收盘复盘.bat collect
REM 用法 2（任务计划每天 15:05 收盘自动回放检验买手/卖手预测准确率并导出报告）：
REM   盘中自动收集与收盘复盘.bat eval
REM 可选：把 PUSH 改成 serverchan:SENDKEY 或 pushplus:TOKEN，收盘复盘结果自动推送到手机

set "PROJ=%~dp0.."
cd /d "%PROJ%"
set "MODE=%~1"
if "%MODE%"=="" set "MODE=collect"
set "PUSH="

if /I "%MODE%"=="eval" (
    echo [1/2] 正在严格因果回放今日及历史买手/卖手盘口快照...
    if "%PUSH%"=="" (
        python stock_predictor.py --eval-orderbook --eval-horizon 60 --json monitor_log\orderbook_eval_latest.json
    ) else (
        python stock_predictor.py --eval-orderbook --eval-horizon 60 --json monitor_log\orderbook_eval_latest.json --push "%PUSH%"
    )
    echo [2/2] 正在验证到期历史预测...
    python stock_predictor.py --verify
    pause
    exit /b 0
)

echo 正在启动手机看盘与盘中五档买手/卖手自动落盘采集（默认每 10 秒轮询，休市自动降频）...
if "%PUSH%"=="" (
    python stock_predictor.py --web --web-port 8765 --web-interval 10
) else (
    python stock_predictor.py --web --web-port 8765 --web-interval 10 --push "%PUSH%"
)
