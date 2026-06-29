#!/usr/bin/env pwsh
#Requires -Version 5.1
<#
.SYNOPSIS
  CUDACyclone — Setup & Build (Windows)
  Detecta dependencias (CUDA Toolkit, Visual Studio, make),
  configura o ambiente e builda o projeto.
#>

$ErrorActionPreference = 'Stop'

Write-Host "+==========================================+" -ForegroundColor DarkCyan
Write-Host "|     CUDACyclone - Setup & Build         |" -ForegroundColor DarkCyan
Write-Host "+==========================================+" -ForegroundColor DarkCyan

function Write-Info   { Write-Host "[INFO]  $args" -ForegroundColor Cyan }
function Write-Ok     { Write-Host "[OK]    $args" -ForegroundColor Green }
function Write-Warn   { Write-Host "[WARN]  $args" -ForegroundColor Yellow }
function Write-Err    { Write-Host "[ERROR] $args" -ForegroundColor Red }

# ── 1. CUDA Toolkit ───────────────────────────
function Find-CudaToolkit {
    # Tenta via PATH primeiro
    $nvccCmd = Get-Command nvcc.exe -ErrorAction SilentlyContinue
    if ($nvccCmd) {
        Write-Ok "nvcc encontrado no PATH"
        $ver = & nvcc.exe --version 2>&1 | Select-String "release"
        if ($ver) {
            if ($ver -match 'release (\S+)') {
                Write-Ok "CUDA Toolkit v$($Matches[1])"
            }
        }
        return $true
    }

    # Varre diretorios do CUDA Toolkit
    $base = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if (Test-Path $base) {
        $cudaDirs = Get-ChildItem -Path "$base\v*" -Directory | Sort-Object Name -Descending
        if ($cudaDirs.Count -gt 0) {
            $latest = $cudaDirs[0].FullName
            Write-Warn "CUDA encontrado em $latest — adicionando ao PATH..."
            $env:Path = "$latest\bin;$latest\libnvvp;$env:Path"
            if (Get-Command nvcc.exe -ErrorAction SilentlyContinue) {
                $ver = & nvcc.exe --version 2>&1 | Select-String "release"
                if ($ver -and $ver -match 'release (\S+)') {
                    Write-Ok "CUDA Toolkit v$($Matches[1]) configurado"
                }
                return $true
            }
        }
    }

    return $false
}

function Install-CudaToolkitWindows {
    Write-Warn "CUDA Toolkit nao encontrado!"
    Write-Warn "Baixe e instale manualmente de:"
    Write-Warn "  https://developer.nvidia.com/cuda-downloads"
    Write-Host ""
    Write-Host "  Tentar instalar via winget? (s/N) " -ForegroundColor Cyan -NoNewline
    $resp = Read-Host
    if ($resp -eq 's' -or $resp -eq 'S') {
        try {
            $proc = Start-Process -FilePath "winget" -ArgumentList "install", "NVIDIA.CUDA", "--accept-package-agreements", "--accept-source-agreements" -Wait -PassThru -NoNewWindow
            if ($proc.ExitCode -eq 0) {
                return (Find-CudaToolkit)
            }
        } catch {
            Write-Err "Falha: $_"
        }
    }
    return $false
}

# ── 2. Visual Studio ──────────────────────────
function Find-VisualStudio {
    $vsPaths = @(
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvars64.bat"
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
    )

    foreach ($p in $vsPaths) {
        if (Test-Path $p) {
            Write-Ok "Visual Studio: $p"
            return $p
        }
    }

    # vswhere
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $vsPath = & $vswhere -latest -property installationPath 2>$null
        if ($vsPath) {
            $vcvars = "$vsPath\VC\Auxiliary\Build\vcvars64.bat"
            if (Test-Path $vcvars) {
                Write-Ok "Visual Studio via vswhere: $vcvars"
                return $vcvars
            }
        }
    }

    # cl.exe ja no PATH?
    if (Get-Command cl.exe -ErrorAction SilentlyContinue) {
        Write-Ok "cl.exe encontrado no PATH"
        return $null
    }

    Write-Warn "Visual Studio / Build Tools nao encontrados."
    Write-Warn "O nvcc precisa do compilador host (cl.exe)."
    Write-Warn "Veja: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022"
    Write-Warn "Selecione a workload 'Desktop development with C++'."
    return $null
}

# ── 3. make ───────────────────────────────────
function Find-Make {
    if (Get-Command make.exe -ErrorAction SilentlyContinue) {
        Write-Ok "make encontrado: $(Get-Command make.exe).Source"
        return $true
    }

    # Procura no PATH expandido
    foreach ($p in $env:Path.Split(';')) {
        $test = Join-Path -Path $p -ChildPath "make.exe"
        if (Test-Path $test) {
            Write-Ok "make encontrado em: $test"
            return $true
        }
    }

    # Chocolatey
    if (Get-Command choco.exe -ErrorAction SilentlyContinue) {
        Write-Warn "Instalar make via Chocolatey? (s/N) " -NoNewline
        $resp = Read-Host
        if ($resp -eq 's' -or $resp -eq 'S') {
            Start-Process -FilePath "choco" -ArgumentList "install", "make", "-y" -Wait -PassThru -NoNewWindow | Out-Null
            if (Get-Command make.exe -ErrorAction SilentlyContinue) {
                Write-Ok "make instalado"
                return $true
            }
        }
    }

    # winget
    if (Get-Command winget.exe -ErrorAction SilentlyContinue) {
        Write-Warn "Instalar make via winget? (s/N) " -NoNewline
        $resp = Read-Host
        if ($resp -eq 's' -or $resp -eq 'S') {
            Start-Process -FilePath "winget" -ArgumentList "install", "GnuWin32.Make", "--accept-package-agreements", "--accept-source-agreements" -Wait -PassThru -NoNewWindow | Out-Null
            # PATH pode nao ter atualizado — varre locais comuns
            $gnuPaths = @(
                "${env:ProgramFiles}\GnuWin32\bin\make.exe",
                "${env:ProgramFiles(x86)}\GnuWin32\bin\make.exe",
                "${env:ProgramFiles}\make\bin\make.exe",
                "${env:ProgramFiles(x86)}\make\bin\make.exe"
            )
            foreach ($gp in $gnuPaths) {
                if (Test-Path $gp) {
                    Write-Ok "make encontrado em: $gp"
                    $env:Path = "$(Split-Path $gp);$env:Path"
                    return $true
                }
            }
            if (Get-Command make.exe -ErrorAction SilentlyContinue) {
                Write-Ok "make instalado"
                return $true
            }
            Write-Warn "make instalado, mas nao encontrado no PATH."
            Write-Warn "Adicione manualmente ao PATH ou use: choco install make"
            return $false
        }
    }

    Write-Warn "make nao encontrado. Instale manualmente: choco install make"
    return $false
}

# ── 4. nvidia-smi ─────────────────────────────
function Find-NvidiaSmi {
    if (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue) {
        $info = & nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>&1
        if ($info) {
            Write-Ok "GPU: $info"
            return $true
        }
    }
    Write-Warn "nvidia-smi nao encontrado (drivers NVIDIA podem nao estar instalados)"
    return $false
}

# ── Detecta compute capability ────────────────
function Get-GpuArch {
    $info = & nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>$null
    if ($info) {
        $cc = $info.Trim() -replace '\.', ''
        if ($cc) { return $cc }
    }
    # Fallback: tenta via nvcc
    $nvccPath = (Get-Command nvcc.exe -ErrorAction SilentlyContinue).Source
    if (-not $nvccPath) { return "86" }  # fallback generico
    return "86"
}

function Get-GpuArchFlags {
    $cc = Get-GpuArch
    $archs = @(75, 86, 89, [int]$cc) | Sort-Object -Unique
    $flags = @()
    foreach ($a in $archs) {
        $flags += "-gencode", "arch=compute_${a},code=sm_${a}"
    }
    Write-Info "Arquiteturas GPU alvo: $($archs -join ', ')"
    return $flags
}

# ── Build com make ────────────────────────────
function Build-WithMake {
    Write-Info "Buildando com make..."
    # Detecta GPU_ARCH antes de qualquer chamada ao make
    # pra evitar que o Makefile tente rodar nvidia-smi via cmd.exe
    $gpuArch = Get-GpuArch
    $env:GPU_ARCH = $gpuArch
    Write-Info "GPU_ARCH=$gpuArch (passado ao make)"
    & make clean 2>&1 | Out-Null
    & make -j $env:NUMBER_OF_PROCESSORS
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "+==========================================+" -ForegroundColor Green
        Write-Host         "|    BUILD CONCLUIDO COM SUCESSO!         |" -ForegroundColor Green
        Write-Host "+==========================================+" -ForegroundColor Green
        Write-Host ""
        $bin = Get-ChildItem "CUDACyclone.exe" -ErrorAction SilentlyContinue
        if ($bin) { Write-Host "  Binario: $($bin.FullName)" }
        Write-Host ""
        Write-Host "  Para executar:"
        Write-Host "    .\CUDACyclone.exe --range INICIO:FIM --address ENDERECO"
        Write-Host ""
        return $true
    }
    return $false
}

# ── Build direto com nvcc ─────────────────────
function Build-WithNvcc {
    Write-Warn "make nao disponivel. Compilando direto com nvcc..."

    # Prepara flags base
    $archFlags = Get-GpuArchFlags
    $commonFlags = @(
        '-O3', '-rdc=true', '-use_fast_math', '--ptxas-options=-O3',
        '-allow-unsupported-compiler'
    ) + $archFlags + @(
        '-std=c++17'
    )
    $linkFlags = @('-lcudadevrt', '-cudart=static')

    $srcs = @('CUDACyclone.cu', 'CUDAHash.cu')
    $objs = @()

    foreach ($s in $srcs) {
        $obj = $s -replace '\.cu$', '.o'
        Write-Info "Compilando $s -> $obj ..."
        & nvcc -c $commonFlags $s -o $obj
        if ($LASTEXITCODE -ne 0) {
            Write-Err "Falha ao compilar $s"
            return $false
        }
        $objs += $obj
    }

    Write-Info "Linkando..."
    & nvcc $commonFlags $objs -o "CUDACyclone.exe" $linkFlags
    if ($LASTEXITCODE -eq 0) {
        Write-Ok "Build concluido! Binario: CUDACyclone.exe"
        return $true
    } else {
        Write-Err "Falha no link. Verifique se o Visual Studio esta configurado corretamente."
        return $false
    }
}

# ── Main ───────────────────────────────────────
function Main {
    Write-Host ""

    # 1. CUDA
    Write-Info "Verificando CUDA Toolkit..."
    if (-not (Find-CudaToolkit)) {
        if (-not (Install-CudaToolkitWindows)) {
            Write-Err "CUDA Toolkit e obrigatorio. Abortando."
            exit 1
        }
    }

    # 2. Visual Studio
    Write-Info "Verificando Visual Studio..."
    $vcvars = Find-VisualStudio
    if ($vcvars) {
        Write-Info "Configurando ambiente via: $vcvars"
        # Executa vcvars64.bat e captura variaveis de ambiente
        $output = & cmd /c "call `"$vcvars`" > nul 2>&1 && set" 2>&1
        foreach ($line in $output) {
            if ($line -match '^([^=]+)=(.*)') {
                $varName = $Matches[1]
                $varValue = $Matches[2]
                Set-Item -Path "env:$varName" -Value $varValue -ErrorAction SilentlyContinue
            }
        }
        Write-Ok "Ambiente Visual Studio configurado"
    }

    # 3. make
    Write-Info "Verificando make..."
    $hasMake = Find-Make

    # 4. GPU
    Write-Info "Verificando GPU NVIDIA..."
    Find-NvidiaSmi | Out-Null

    # 5. Build
    Write-Host ""
    Write-Info "Iniciando build..."
    Write-Host ""

    if (Test-Path "Makefile") {
        if ($hasMake) {
            if (-not (Build-WithMake)) {
                Write-Err "Build com make falhou."
                exit 1
            }
        } else {
            if (-not (Build-WithNvcc)) {
                Write-Err "Build com nvcc falhou."
                exit 1
            }
        }
    } else {
        Write-Err "Makefile nao encontrado."
        exit 1
    }
}

Main
