param(
    [switch]$Clean,
    [switch]$SkipPyCompile,
    [switch]$SkipInstaller
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Resolve-Python {
    $venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        return $venvPython
    }

    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCmd) {
        return $pythonCmd.Source
    }

    throw "Python nao encontrado. Ative a .venv ou instale Python antes do build."
}

function Resolve-Iscc {
    $cmd = Get-Command ISCC.exe -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }

    $candidatos = @(
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "${env:ProgramFiles}\Inno Setup 6\ISCC.exe"
    )

    foreach ($candidato in $candidatos) {
        if ($candidato -and (Test-Path $candidato)) {
            return $candidato
        }
    }

    throw "ISCC.exe nao encontrado. Instale o Inno Setup 6 para gerar o instalador."
}

function Invoke-Checked {
    param(
        [string]$FilePath,
        [string[]]$Arguments
    )

    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Falha ao executar: $FilePath $($Arguments -join ' ')"
    }
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
Set-Location $projectRoot

$pythonExe = Resolve-Python
$isccExe = $null
$specPath = Join-Path $scriptDir "loteria.spec"
$installerPath = Join-Path $scriptDir "installer.iss"

Write-Step "Validando arquivos obrigatorios"
$requiredFiles = @(
    "main.py",
    "ui.py",
    "predicao.py",
    "coleta.py",
    "app_paths.py",
    "loteria.ico",
    "requirements.txt"
)

foreach ($file in $requiredFiles) {
    if (-not (Test-Path (Join-Path $projectRoot $file))) {
        throw "Arquivo obrigatorio ausente: $file"
    }
}

foreach ($file in @($specPath, $installerPath)) {
    if (-not (Test-Path $file)) {
        throw "Arquivo obrigatorio ausente: $file"
    }
}

if (-not (Test-Path (Join-Path $projectRoot "modelos"))) {
    throw "Pasta 'modelos' nao encontrada."
}

if ($Clean) {
    Write-Step "Limpando build anterior"
    foreach ($path in @("build", "dist")) {
        $fullPath = Join-Path $projectRoot $path
        if (Test-Path $fullPath) {
            Remove-Item -LiteralPath $fullPath -Recurse -Force
        }
    }
}

Write-Step "Conferindo PyInstaller"
try {
    Invoke-Checked -FilePath $pythonExe -Arguments @("-m", "PyInstaller", "--version")
}
catch {
    Write-Step "PyInstaller nao encontrado, instalando"
    try {
        Invoke-Checked -FilePath $pythonExe -Arguments @("-m", "pip", "install", "pyinstaller")
    }
    catch {
        throw "Nao foi possivel instalar o PyInstaller automaticamente. Instale manualmente no Python ativo e rode o script novamente."
    }
}

if (-not $SkipPyCompile) {
    Write-Step "Validando sintaxe"
    Invoke-Checked -FilePath $pythonExe -Arguments @(
        "-m",
        "py_compile",
        "app_paths.py",
        "coleta.py",
        "predicao.py",
        "ui.py",
        "main.py"
    )
}

Write-Step "Gerando executavel"
Invoke-Checked -FilePath $pythonExe -Arguments @("-m", "PyInstaller", "--clean", $specPath)

if ($SkipInstaller) {
    Write-Step "Build finalizado sem instalador"
    Write-Host "Executavel disponivel em: dist\PrevisorLoteriasInteligente\PrevisorLoteriasInteligente.exe" -ForegroundColor Green
    exit 0
}

$isccExe = Resolve-Iscc

Write-Step "Gerando instalador"
Invoke-Checked -FilePath $isccExe -Arguments @($installerPath)

Write-Step "Build concluido"
Write-Host "Executavel: dist\PrevisorLoteriasInteligente\PrevisorLoteriasInteligente.exe" -ForegroundColor Green
Write-Host "Instalador: dist\installer\PrevisorLoteriasInteligente-Setup.exe" -ForegroundColor Green
