@echo off
echo Setting up Git repository for ML Trading Application
echo =================================================

cd "d:\workspace\ML_Automated_Trading_Robinhood-master\ML_Automated_Trading_Robinhood-master"

echo Initializing Git repository...
git init

echo Adding all files to staging...
git add .

echo Checking status...
git status

echo Committing changes...
git commit -m "Initial commit: Enhanced ML Trading Application with optimized parameters and comprehensive analysis"

echo Setting main branch...
git branch -M main

echo Adding remote repository...
git remote add origin https://github.com/chakrapani210/MLTradingApp.git

echo Pushing to GitHub...
git push -u origin main

echo.
echo =================================================
echo Git setup complete! 
echo Repository should now be available at:
echo https://github.com/chakrapani210/MLTradingApp
echo =================================================

pause