# ── Detectar plataforma ──────────────────────────
ifeq ($(OS),Windows_NT)
  TARGET_EXT := .exe
  OBJ_EXT    := .obj
  RM         := cmd /c del /Q /F 2>nul
else
  TARGET_EXT :=
  OBJ_EXT    := .o
  RM         := rm -f
endif

TARGET      := CUDACyclone$(TARGET_EXT)
SRC         := CUDACyclone.cu CUDAHash.cu
OBJ         := $(SRC:.cu=$(OBJ_EXT))
CC          := nvcc

# Detecta compute capability de TODAS as GPUs instaladas
# Windows: usa PowerShell (consegue acessar nvidia-smi no PATH)
# Linux: usa shell normal
ifeq ($(OS),Windows_NT)
  GPU_ARCHS ?= $(sort $(strip $(shell powershell -NoProfile -Command "try{ (nvidia-smi --query-gpu=compute_cap --format=csv,noheader) -replace '\.','' -join ' ' }catch{''}" 2>nul)))
else
  GPU_ARCHS ?= $(sort $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sed 's/\.//' | tr '\n' ' '))
endif
ifeq ($(GPU_ARCHS),)
  GPU_ARCHS := 86
  $(warning nvidia-smi nao disponivel. Usando GPU_ARCHS=86 como fallback.)
endif

SM_ARCHS   := $(sort 75 86 89 $(GPU_ARCHS))
GENCODE    := $(foreach arch,$(SM_ARCHS),-gencode arch=compute_$(arch),code=sm_$(arch))

NVCC_FLAGS := -O3 -rdc=true -use_fast_math --ptxas-options=-O3 -allow-unsupported-compiler $(GENCODE)
CXXFLAGS   := -std=c++17

LDFLAGS    := -lcudadevrt -cudart=static

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(NVCC_FLAGS) $(CXXFLAGS) $(OBJ) -o $@ $(LDFLAGS)

%$(OBJ_EXT): %.cu
	$(CC) $(NVCC_FLAGS) $(CXXFLAGS) -c $< -o $@

clean:
	-$(RM) $(TARGET) $(OBJ)

