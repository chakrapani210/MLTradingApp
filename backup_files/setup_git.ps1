# PowerShell script to setup Git repository
Write-Host "Setting up Git repository for ML Trading Application" -ForegroundColor Green
Write-Host "=================================================" -ForegroundColor Green

# Navigate to project directory
Set-Location "d:\workspace\ML_Automated_Trading_Robinhood-master\ML_Automated_Trading_Robinhood-master"

# Initialize Git repository
Write-Host "Initializing Git repository..." -ForegroundColor Yellow
git init

# Add all files
Write-Host "Adding files to staging..." -ForegroundColor Yellow
git add .

# Check status
Write-Host "Checking status..." -ForegroundColor Yellow
git status

# Commit changes
Write-Host "Committing changes..." -ForegroundColor Yellow
git commit -m "Initial commit: Enhanced ML Trading Application with optimized parameters and comprehensive analysis"

# Set main branch
Write-Host "Setting main branch..." -ForegroundColor Yellow
git branch -M main

# Add remote
Write-Host "Adding remote repository..." -ForegroundColor Yellow
git remote add origin https://github.com/chakrapani210/MLTradingApp.git

# Push to GitHub
Write-Host "Pushing to GitHub..." -ForegroundColor Yellow
git push -u origin main

Write-Host ""
Write-Host "=================================================" -ForegroundColor Green
Write-Host "Git setup complete!" -ForegroundColor Green
Write-Host "Repository should now be available at:" -ForegroundColor Green
Write-Host "https://github.com/chakrapani210/MLTradingApp" -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Green

Read-Host "Press Enter to continue..."