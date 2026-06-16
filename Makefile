CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -pedantic -pthread

NVCC ?= nvcc
NVCCFLAGS ?= -O3 -std=c++17 -arch=native

SRC_DIR := src
SRCS := $(SRC_DIR)/main_gravity_wave.cpp \
        $(SRC_DIR)/Params.cpp \
        $(SRC_DIR)/Camera.cpp \
        $(SRC_DIR)/Image.cpp \
        $(SRC_DIR)/ModeData.cpp \
        $(SRC_DIR)/SpinWeightedHarmonic.cpp \
        $(SRC_DIR)/GravityWaveField.cpp \
        $(SRC_DIR)/BuildScene.cpp \
        $(SRC_DIR)/Renderer.cpp

OBJS := $(SRCS:.cpp=.o)
TARGET := main_gravity_wave

CUDA_CPP_SRCS := $(SRC_DIR)/main_gravity_wave_cuda.cpp \
                 $(SRC_DIR)/Params.cpp \
                 $(SRC_DIR)/Camera.cpp \
                 $(SRC_DIR)/Image.cpp \
                 $(SRC_DIR)/ModeData.cpp \
                 $(SRC_DIR)/SpinWeightedHarmonic.cpp \
                 $(SRC_DIR)/GravityWaveField.cpp \
                 $(SRC_DIR)/BuildScene.cpp \
                 $(SRC_DIR)/NumericalMetricData.cpp

LOOKINGGLASS_CPP_SRCS := $(SRC_DIR)/main_gravity_wave_cude_lookingglass.cpp \
                         $(SRC_DIR)/Params.cpp \
                         $(SRC_DIR)/Camera.cpp \
                         $(SRC_DIR)/Image.cpp \
                         $(SRC_DIR)/ModeData.cpp \
                         $(SRC_DIR)/SpinWeightedHarmonic.cpp \
                         $(SRC_DIR)/GravityWaveField.cpp \
                         $(SRC_DIR)/BuildScene.cpp \
                         $(SRC_DIR)/NumericalMetricData.cpp

CUDA_CU_SRCS := $(SRC_DIR)/RendererCuda.cu

CUDA_CPP_OBJS := $(CUDA_CPP_SRCS:%.cpp=%.cuda.o)
LOOKINGGLASS_CPP_OBJS := $(LOOKINGGLASS_CPP_SRCS:%.cpp=%.lookingglass.o)
CUDA_CU_OBJS := $(CUDA_CU_SRCS:.cu=.cu.o)

CUDA_TARGET := main_gravity_wave_cuda
LOOKINGGLASS_TARGET := main_gravity_wave_cude_lookingglass

PYTHON ?= python
DATADIR ?= gw150914_data/GW150914_28
MODE_EXPORT ?= modes_psi4_l4.txt
PANORAMA ?= panorama.jpeg

.PHONY: all cuda lookingglass clean export preview render

all: $(TARGET)

cuda: $(CUDA_TARGET)

lookingglass: $(LOOKINGGLASS_TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

$(CUDA_TARGET): $(CUDA_CPP_OBJS) $(CUDA_CU_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $(CUDA_CPP_OBJS) $(CUDA_CU_OBJS)

$(LOOKINGGLASS_TARGET): $(LOOKINGGLASS_CPP_OBJS) $(CUDA_CU_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $(LOOKINGGLASS_CPP_OBJS) $(CUDA_CU_OBJS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.cuda.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.lookingglass.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.cu.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

export:
	$(PYTHON) export_psi4_modes_to_text.py --datadir $(DATADIR) --out $(MODE_EXPORT) --max-l 4

preview: all
	./$(TARGET) --modes-file $(MODE_EXPORT) --panorama $(PANORAMA) --out gravity_wave_volume_preview.png --width 640 --height 640 --spp 1 --step 3 --wave-radius 420 --harmonics "2,2 2,-2"

render: all
	./$(TARGET) --modes-file $(MODE_EXPORT) --panorama $(PANORAMA) --out gravity_wave_volume.png --width 1000 --height 1000 --spp 1 --step 2 --wave-radius 420 --harmonics "2,2 2,-2"

clean:
	rm -f $(OBJS) $(TARGET) $(CUDA_CPP_OBJS) $(LOOKINGGLASS_CPP_OBJS) $(CUDA_CU_OBJS) $(CUDA_TARGET) $(LOOKINGGLASS_TARGET)
