#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────
#  CUDACyclone — Setup & Build (Linux)
#  Detecta dependências, instala o que faltar,
#  e builda o projeto automaticamente.
# ──────────────────────────────────────────────

RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*"; }

# ── Detect distro ─────────────────────────────
detect_distro() {
    if   [ -f /etc/os-release ]; then . /etc/os-release; echo "$ID"
    elif [ -f /etc/debian_version ]; then echo "debian"
    elif [ -f /etc/redhat-release ]; then echo "rhel"
    elif command -v pacman &>/dev/null; then echo "arch"
    else echo "unknown"; fi
}

DISTRO=$(detect_distro)
info "Distro detectada: ${DISTRO}"

# ── Verifica / instala CUDA toolkit ────────────
check_cuda() {
    if command -v nvcc &>/dev/null; then
        CUDA_VER=$(nvcc --version | grep "release" | sed -E 's/.*release ([0-9]+\.[0-9]+).*/\1/')
        ok "CUDA Toolkit ${CUDA_VER} encontrado em $(which nvcc)"
        return 0
    fi
    return 1
}

install_cuda_debian() {
    warn "CUDA Toolkit não encontrado. Instalando via NVIDIA package manager..."
    # Pega a versão do Ubuntu/Debian
    UBUNTU_VER=$(lsb_release -rs 2>/dev/null || echo "22.04")
    # Monta a URL do repositório NVIDIA
    local distro_arch="x86_64"
    local os_name="ubuntu$(echo "$UBUNTU_VER" | tr -d '.')"
    
    info "Baixando CUDA keyring..."
    wget -q "https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}${UBUNTU_VER}/${distro_arch}/cuda-keyring_1.1-1_all.deb" -O /tmp/cuda-keyring.deb 2>/dev/null || {
        # Fallback: método alternativo
        apt-get update -qq
        DEBIAN_FRONTEND=noninteractive apt-get install -y -qq nvidia-cuda-toolkit 2>&1 | tail -5
        if command -v nvcc &>/dev/null; then
            ok "CUDA instalado via apt (nvidia-cuda-toolkit)"
            return 0
        fi
        err "Falha ao instalar CUDA. Instale manualmente:"
        err "  https://developer.nvidia.com/cuda-downloads"
        return 1
    }
    dpkg -i /tmp/cuda-keyring.deb && apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cuda-toolkit 2>&1 | tail -5
    rm -f /tmp/cuda-keyring.deb
}

install_cuda_rhel() {
    warn "CUDA Toolkit não encontrado. Instalando..."
    if command -v dnf &>/dev/null; then
        dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel$(rpm -E %rhel)/x86_64/cuda-rhel$(rpm -E %rhel).repo
        dnf install -y cuda-toolkit 2>&1 | tail -5
    else
        yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel$(rpm -E %rhel)/x86_64/cuda-rhel$(rpm -E %rhel).repo
        yum install -y cuda-toolkit 2>&1 | tail -5
    fi
}

install_cuda_arch() {
    warn "CUDA Toolkit não encontrado. Instalando via pacman..."
    pacman -Sy --noconfirm cuda 2>&1 | tail -5
}

install_cuda_auto() {
    case "$DISTRO" in
        debian|ubuntu) install_cuda_debian ;;
        rhel|fedora|centos) install_cuda_rhel ;;
        arch) install_cuda_arch ;;
        *)
            err "Distro '$DISTRO' não suportada para instalação automática."
            err "Instale o CUDA Toolkit manualmente: https://developer.nvidia.com/cuda-downloads"
            return 1
            ;;
    esac
}

# ── Verifica / instala build tools ────────────
check_build_tools() {
    local missing=0
    for cmd in make g++; do
        if ! command -v "$cmd" &>/dev/null; then
            warn "$cmd não encontrado."
            missing=1
        fi
    done
    if [ "$missing" -eq 0 ]; then
        ok "Build tools (make, g++) disponíveis"
        return 0
    fi
    return 1
}

install_build_tools() {
    warn "Instalando build tools..."
    case "$DISTRO" in
        debian|ubuntu)     apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq build-essential 2>&1 | tail -3 ;;
        rhel|fedora|centos) dnf install -y gcc-c++ make 2>&1 | tail -3 || yum install -y gcc-c++ make 2>&1 | tail -3 ;;
        arch)              pacman -Sy --noconfirm base-devel 2>&1 | tail -3 ;;
        *)                 err "Não foi possível instalar build tools automaticamente."; return 1 ;;
    esac
}

# ── Adiciona CUDA ao PATH se necessário ────────
ensure_cuda_in_path() {
    # Tenta achar nvcc em lugares comuns
    if ! command -v nvcc &>/dev/null; then
        for dir in /usr/local/cuda-*/bin /opt/cuda/bin /usr/lib/cuda/bin; do
            if [ -f "$dir/nvcc" ]; then
                export PATH="$dir:$PATH"
                ok "CUDA encontrado em $dir (adicionado ao PATH)"
                break
            fi
        done
        # Fallback: /usr/local/cuda (symlink mais recente)
        if ! command -v nvcc &>/dev/null && [ -f /usr/local/cuda/bin/nvcc ]; then
            export PATH="/usr/local/cuda/bin:$PATH"
            ok "CUDA encontrado em /usr/local/cuda/bin"
        fi
    fi
    
    # Biblioteca CUDA pro linker
    if [ -d /usr/local/cuda/lib64 ]; then
        export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
    fi
    
    if ! command -v nvcc &>/dev/null; then
        err "nvcc não encontrado. CUDA Toolkit não está instalado."
        return 1
    fi
}

# ── Main ───────────────────────────────────────
main() {
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║     CUDACyclone — Setup & Build         ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════╝${NC}"
    echo ""
    
    # 1. CUDA Toolkit
    info "Verificando CUDA Toolkit..."
    if ! check_cuda; then
        ensure_cuda_in_path || install_cuda_auto || { err "CUDA é obrigatório. Abortando."; exit 1; }
        check_cuda || { err "CUDA ainda não detectado após instalação. Abortando."; exit 1; }
    fi
    
    # 2. Build tools
    info "Verificando build tools..."
    check_build_tools || { install_build_tools; check_build_tools || { err "Build tools não disponíveis. Abortando."; exit 1; } }
    
    # 3. GPU / driver
    info "Verificando GPU NVIDIA..."
    GPU_ARCH="86"  # fallback padrão
    if command -v nvidia-smi &>/dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
        GPU_ARCH=$(echo "$GPU_CC" | tr -d '.')
        ok "GPU: ${GPU_NAME} (Compute ${GPU_CC})"
    else
        warn "nvidia-smi não encontrado. Drivers NVIDIA podem não estar instalados."
        warn "O binário será compilado mas pode não executar sem GPU NVIDIA + driver."
    fi
    export GPU_ARCH
    ok "GPU_ARCH=${GPU_ARCH} (passado ao make)"
    
    # 4. Build!
    echo ""
    info "Buildando o projeto..."
    make clean 2>/dev/null || true
    if make -j$(nproc); then
        echo ""
        echo -e "${GREEN}╔══════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║  ✅  BUILD CONCLUÍDO COM SUCESSO!       ║${NC}"
        echo -e "${GREEN}╚══════════════════════════════════════════╝${NC}"
        echo ""
        echo -e "  Binário: ${CYAN}$(pwd)/CUDACyclone${NC}"
        echo -e "  Tamanho: $(ls -lh CUDACyclone | awk '{print $5}')"
        echo ""
        echo -e "  Para executar:"
        echo -e "    ${YELLOW}./CUDACyclone --range INICIO:FIM --address ENDERECO${NC}"
        echo ""
    else
        err "Build falhou!"
        exit 1
    fi
}

main "$@"
