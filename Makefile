CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -pedantic -pthread

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

PYTHON ?= python
DATADIR ?= gw150914_data/GW150914_28
MODE_EXPORT ?= modes_psi4_l4.txt
PANORAMA ?= panorama.jpeg

.PHONY: all clean export preview render

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

export:
	$(PYTHON) export_psi4_modes_to_text.py --datadir $(DATADIR) --out $(MODE_EXPORT) --max-l 4

preview: all
	./$(TARGET) --modes-file $(MODE_EXPORT) --panorama $(PANORAMA) --out gravity_wave_volume_preview.png --width 640 --height 640 --spp 1 --step 3 --wave-radius 420 --harmonics "2,2 2,-2"

render: all
	./$(TARGET) --modes-file $(MODE_EXPORT) --panorama $(PANORAMA) --out gravity_wave_volume.png --width 1000 --height 1000 --spp 1 --step 2 --wave-radius 420 --harmonics "2,2 2,-2"

clean:
	rm -f $(OBJS) $(TARGET)
