# E0 准备:安装 ToolMaze、下载数据、建立 8K 上下文的本地模型变体,再跑 5 道 P0 题做管线检查。
# 用法(在 D:\ZSH\Agent 下):双击 scripts\run_e0.bat,或 powershell -ExecutionPolicy Bypass -File scripts\e0_setup.ps1
# 所有输出都写进 external\logs\e0.log,Claude 从那里读取。

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Ext  = Join-Path $Root "external"
$AE   = Join-Path $Ext "agent_error"
$TM   = Join-Path $AE "toolmaze"
$Venv = Join-Path $Ext "venv-bench"
$Py   = Join-Path $Venv "Scripts\python.exe"
$Logs = Join-Path $Ext "logs"
New-Item -ItemType Directory -Force $AE, $Logs | Out-Null
Start-Transcript -Path (Join-Path $Logs "e0.log") -Force | Out-Null

# 读取仓库根目录的 .env(KEY=VALUE,# 开头为注释);密钥只放这里,不进仓库
$EnvFile = Join-Path $Root ".env"
if (Test-Path $EnvFile) {
  Get-Content $EnvFile -Encoding UTF8 | Where-Object { $_ -match '^\s*[A-Za-z_][A-Za-z0-9_]*\s*=' } | ForEach-Object {
    $k, $v = $_ -split '=', 2
    $k = $k.Trim(); $v = $v.Trim().Trim('"')
    if ($v) { Set-Item -Path "Env:$k" -Value $v }
  }
} else { Write-Warning "没有找到 .env:复制 .env.example 为 .env 并填写" }

# 大文件全部放 D 盘:pip 缓存、HuggingFace 缓存
$env:PIP_CACHE_DIR = "D:\ZSH\cache\pip"
$env:HF_HOME       = "D:\ZSH\cache\hf"
# ToolMaze 用 open(f, 'r') 读 JSON 不指定编码;中文 Windows 默认 GBK,会读坏 UTF-8 文件
$env:PYTHONUTF8    = "1"

function Step($name) { Write-Host "`n===== $name =====" }

Step "1. 环境信息"
git --version; python --version; ollama --version
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv

Step "2. 下载 ToolMaze 代码"
if (-not (Test-Path $TM)) { git clone --depth 1 https://github.com/Zhudongsheng75/ToolMaze.git $TM }
git -C $TM log -1 --format="commit %H  %cd"

Step "3. 独立虚拟环境(不影响项目的 venv)"
if (-not (Test-Path $Py)) { python -m venv $Venv }
& $Py -m pip install -q --upgrade pip
& $Py -m pip install -q -r (Join-Path $TM "requirements.txt") huggingface_hub
& $Py -m pip freeze | Out-File (Join-Path $Logs "e0_pip_freeze.txt") -Encoding utf8

Step "4. 下载 ToolMaze 数据集"
$Dl = Join-Path $TM "data_download"
& $Py -c "from huggingface_hub import snapshot_download; p=snapshot_download('dongsheng/ToolMaze', repo_type='dataset', local_dir=r'$Dl'); print(p)"
# 数据集若是压缩包,解压到 data\;若已是目录结构,原样复制
$Data = Join-Path $TM "data"
New-Item -ItemType Directory -Force $Data | Out-Null
Get-ChildItem $Dl -Filter *.zip -Recurse | ForEach-Object { Expand-Archive $_.FullName -DestinationPath $Data -Force }
if (-not (Get-ChildItem $Dl -Filter *.zip -Recurse)) { Copy-Item (Join-Path $Dl "*") $Data -Recurse -Force }
Get-ChildItem $Data -Recurse -Directory | Select-Object -First 30 FullName
foreach ($c in "c1","c2","c3","c4") {
  $d = Join-Path $Data "perturbed_tasks\$c"
  if (Test-Path $d) { Write-Host "$c 任务文件数: $((Get-ChildItem $d -Filter *.json).Count)" }
}

Step "5. 本地模型变体:固定 8192 上下文和推荐采样参数"
# OpenAI 兼容接口(/v1)不能逐请求传 num_ctx,不固定的话 Ollama 用默认上下文,超出部分静默截断
$Mf = Join-Path $Ext "Modelfile.qwen3i-8k"
@"
FROM qwen3:4b-instruct
PARAMETER num_ctx 8192
PARAMETER top_k 20
PARAMETER top_p 0.8
"@ | Out-File $Mf -Encoding ascii
ollama create qwen3i-8k -f $Mf
ollama show qwen3i-8k --parameters

Step "6. 检查 Ollama 的 OpenAI 兼容接口"
Invoke-RestMethod http://localhost:11434/v1/models | ConvertTo-Json -Depth 4

Step "7. 管线检查:C1 前 5 道 P0 题(串行)"
if (-not $env:SILICONFLOW_API_KEY) { Write-Warning ".env 里的 SILICONFLOW_API_KEY 为空,裁判将无法调用" }
$Cfg = Join-Path $Root "eval\agent_error\configs\toolmaze_e0.yaml"
Push-Location $TM
$t0 = Get-Date
& $Py evaluation\scripts\run_eval.py --config $Cfg --modes P0 --limit 5 --no-parallel
Write-Host ("总耗时 {0:N1} 秒" -f ((Get-Date) - $t0).TotalSeconds)
Pop-Location
nvidia-smi --query-gpu=memory.used --format=csv

Step "完成"
Stop-Transcript | Out-Null
