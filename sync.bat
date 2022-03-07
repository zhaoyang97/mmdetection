set sec=0
set rto=1
:loop
    git pull
    git push
    timeout /T %sec%
    set /a sec=sec+rto
goto:loop