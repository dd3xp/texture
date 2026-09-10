# 每 30 分钟拉起一个全新的无头 Claude 干一轮。
# 由 Windows 计划任务 TextureCron 调用。
#
# 为什么是计划任务而不是 Claude Code 的 CronCreate：
# 后者只活在会话内存里、要求 REPL 空闲，本项目实测多次不触发；
# 计划任务独立于会话，会话关了照样跑。

$ErrorActionPreference = 'Continue'
$root   = 'C:\Codes\texture'
$logDir = Join-Path $root 'logs_local'
$lock   = Join-Path $logDir 'cron.lock'
$log    = Join-Path $logDir 'cron.log'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# 单实例：上一轮没跑完就跳过，避免并发改同一个仓库
if (Test-Path $lock) {
    $age = (Get-Date) - (Get-Item $lock).LastWriteTime
    if ($age.TotalMinutes -lt 90) {
        Add-Content -Encoding utf8 $log ("[{0}] previous round still running ({1} min), skip" -f (Get-Date -f 'MM-dd HH:mm'), [int]$age.TotalMinutes)
        exit 0
    }
    Add-Content -Encoding utf8 $log ("[{0}] lock stale ({1} min), taking over" -f (Get-Date -f 'MM-dd HH:mm'), [int]$age.TotalMinutes)
}
# 与交互会话互斥：人（或主会话）刚提交过就跳过这一轮。
# 起因：两个会话并发跑了同一个实验并写出互相矛盾的结论（B16），
# 那次靠无头轮次自己发现才纠正，不能指望每次都如此。
$since = & git -C $root log -1 --format=%ct 2>$null
if ($since) {
    $mins = ([DateTimeOffset]::UtcNow.ToUnixTimeSeconds() - [int]$since) / 60
    if ($mins -lt 20) {
        Add-Content -Encoding utf8 $log ("[{0}] main session active ({1:N0} min since last commit), skip" -f (Get-Date -f 'MM-dd HH:mm'), $mins)
        exit 0
    }
}

Set-Content -Encoding utf8 $lock (Get-Date -f 'o')

try {
    Set-Location $root
    # 计划任务里若继承到会话变量，claude 会拒绝嵌套启动；显式清掉。
    foreach ($v in 'CLAUDECODE','CLAUDE_CODE_ENTRYPOINT','CLAUDE_CODE_SESSION_ID',
                   'CLAUDE_PID','CLAUDE_CODE_MESSAGING_SOCKET',
                   'CLAUDE_CODE_MESSAGING_TOKEN','CLAUDE_CODE_CHILD_SESSION') {
        Remove-Item "env:$v" -ErrorAction SilentlyContinue
    }
    # **计划任务 403 的真因**：代理只存在于交互会话里，不是持久环境变量，
    # 调度器的裸环境拿不到它，claude 连不上 API。
    # 本机代理是 127.0.0.1:7890（无凭据），显式设上。
    if (-not $env:HTTPS_PROXY) { $env:HTTPS_PROXY = 'http://127.0.0.1:7890' }
    if (-not $env:HTTP_PROXY)  { $env:HTTP_PROXY  = 'http://127.0.0.1:7890' }
    if (-not $env:NO_PROXY)    { $env:NO_PROXY    = 'localhost,127.0.0.1' }

    $cred = Join-Path $env:USERPROFILE '.claude\.credentials.json'
    Add-Content -Encoding utf8 $log ("[{0}] diag user={1} home={2} cred={3} claude={4}" -f `
        (Get-Date -f 'MM-dd HH:mm'), $env:USERNAME, $env:USERPROFILE,
        (Test-Path $cred), ((Get-Command claude -ErrorAction SilentlyContinue).Source))
    $stamp  = Get-Date -f 'MMdd_HHmm'
    $out    = Join-Path $logDir "round_$stamp.log"
    $prompt = Get-Content -Raw -Encoding utf8 (Join-Path $root 'scripts\cron_prompt.md')

    Add-Content -Encoding utf8 $log ("[{0}] round start -> round_{1}.log" -f (Get-Date -f 'MM-dd HH:mm'), $stamp)

    # 提示词走**标准输入**，不走命令行参数：
    # PowerShell 把多行字符串按空白拆成多个参数传给原生 exe，
    # 提示词里的 `tail -3` 会被 claude 当成未知选项而直接失败。
    # **必须显式给 --model**：不给就继承用户配置里的默认值，而那个值（`fable[1m]`）
    # 在无头会话里取不到，claude 一句 "issue with the selected model" 就退出，
    # 整轮什么也没干却记成 "round done"（09-10 05:07 那轮即如此）。
    $prompt | & claude -p --dangerously-skip-permissions --output-format text `
        --model claude-opus-5 --max-turns 60 2>&1 | Out-File -Encoding utf8 $out

    $tail = (Get-Content $out -Tail 2 -ErrorAction SilentlyContinue) -join ' '
    $head = (git -C $root log --oneline -1) 2>$null
    $t2 = $tail -replace '\s+', ' '
    Add-Content -Encoding utf8 $log ("[{0}] round done | HEAD {1} | tail {2}" -f (Get-Date -f 'MM-dd HH:mm'), ${head}, ${t2})
}
catch {
    Add-Content -Encoding utf8 $log ("[{0}] ERROR {1}" -f (Get-Date -f 'MM-dd HH:mm'), $_.Exception.Message)
}
finally {
    Remove-Item $lock -ErrorAction SilentlyContinue
}
