<#
Creates/updates a Windows Scheduled Task that runs the scheduled_trading_task.py every weekday at 2:30 PM Central Time.
Assumptions:
- Python is on PATH (else adjust $PythonExe)
- Repo path is this script's parent directory
- Task runs even if user is logged in (adjust as needed)
#>
param(
    [string]$TaskName = 'MLTrading_Scheduled_1430CST',
    [string]$Time = '14:30' # local machine time (ensure machine clock is Central or adjust for offset)
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
$PythonExe = 'python'
$ScriptPath = Join-Path $RepoRoot 'scripts\scheduled_trading_task.py'

if (!(Test-Path $ScriptPath)) { Write-Error "Script not found: $ScriptPath"; exit 1 }

# Create action
$Action = New-ScheduledTaskAction -Execute $PythonExe -Argument "`"$ScriptPath`""
# Daily trigger (you can restrict to Mon-Fri using -DaysOfWeek)
$Trigger = New-ScheduledTaskTrigger -Daily -At ($Time)
# Optionally restrict to weekdays:
# $Trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At ($Time)

# Register or update
try {
    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    }
    Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Description 'ML Trading daily scheduled run at 14:30 CST' -RunLevel Highest
    Write-Host "Scheduled task '$TaskName' created to run $ScriptPath at $Time (local)." -ForegroundColor Green
}
catch {
    Write-Error $_
}
