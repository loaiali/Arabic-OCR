echo off
set logFile=runs.log
set sep=-----------------------------------

for /f "usebackq tokens=1*" %%i in (`echo %*`) DO @ set params=%%j
set module=%1

echo %DATE% %sep% %module% (%params%): %sep% >> %logFile%
python %module% %params% >> %logFile%