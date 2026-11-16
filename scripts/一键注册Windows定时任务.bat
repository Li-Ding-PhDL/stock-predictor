@echo off
chcp 65001 >nul
REM ============ 一键注册 Windows 任务计划（工作日自动采集与收盘复盘） ============
REM 运行后将在 Windows 任务计划程序里注册两个工作日任务：
REM   1) StockPredictor_Collect_0928 : 周一至周五 09:28 启动盘中自动采集+手机看盘服务
REM   2) StockPredictor_Eval_1505    : 周一至周五 15:05 执行买手/卖手盘口回放准确率评估与到期核验
REM 如需卸载这两个定时任务，请传参数 remove：
REM   一键注册Windows定时任务.bat remove

set "SCRIPT_DIR=%~dp0"
set "BAT_PATH=%SCRIPT_DIR%盘中自动收集与收盘复盘.bat"

if /I "%~1"=="remove" (
    schtasks /Delete /TN "StockPredictor_Collect_0928" /F >nul 2>&1
    schtasks /Delete /TN "StockPredictor_Eval_1505" /F >nul 2>&1
    echo 已移除定时任务 StockPredictor_Collect_0928 与 StockPredictor_Eval_1505。
    pause
    exit /b 0
)

schtasks /Create /TN "StockPredictor_Collect_0928" /TR "\"%BAT_PATH%\" collect" /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 09:28 /F
schtasks /Create /TN "StockPredictor_Eval_1505" /TR "\"%BAT_PATH%\" eval" /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 15:05 /F

echo.
echo [完成] 已尝试注册以下两个工作日定时任务：
echo   - StockPredictor_Collect_0928 (周一~周五 09:28 盘中自动采集)
echo   - StockPredictor_Eval_1505    (周一~周五 15:05 收盘准确率回放)
echo 如提示权限不足，请右键本脚本选择“以管理员身份运行”。
pause
