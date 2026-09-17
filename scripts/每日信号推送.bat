@echo off
chcp 65001 >nul
REM ============ 暴跌抄反弹·每日信号自动推送到手机 ============
REM 用法：把下面 PUSH 改成你自己的免费推送 token，然后用 Windows「任务计划程序」每天 15:35 运行本 .bat。
REM   微信推送(推荐)：先去 https://www.pushplus.plus 注册(免费)、扫码绑定微信、拿到 token，填到 PUSH。
REM   或方糖 Server酱：https://sct.ftqq.com 拿 SENDKEY，PUSH 改成 serverchan:你的SENDKEY
REM 不填 PUSH 也行——只生成手机友好 HTML(在项目目录 每日信号_日期.html)，自己发到手机看。

set PROJ=C:\Users\Q\Desktop\py\stock-predictor(1)\stock-predictor
set PUSH=pushplus:在这里填你的TOKEN
set STOCK_LOCAL_DATA_ROOT=%PROJ%\data_updated

cd /d "%PROJ%"
REM 先把数据更新到最新交易日(baostock 免限流)，再扫今日买点并推送
python stock_predictor.py --cli --update-data --global-scope mine
python stock_predictor.py --cli --dip-daily --global-scope mine --dip-n 10 --dip-x 8 --dip-y 5 --push "%PUSH%"
