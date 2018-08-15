VULKAN_SDK_PATH = /home/dcastrop/Vulkan/1.1.82.0/x86_64
CFLAGS = -std=c++11 -I$(VULKAN_SDK_PATH)/include
LDFLAGS = -L$(VULKAN_SDK_PATH)/lib `pkg-config --static --libs glfw3` -lvulkan

computeConstant: computeConstant.cpp shaders/vert.spv shaders/frag.spv
	g++ $(CFLAGS) -o computeConstant computeConstant.cpp $(LDFLAGS)

shaders/vert.spv: shaders/shader.vert
	glslangValidator -V shaders/shader.vert -o shaders/vert.spv

shaders/frag.spv: shaders/shader.frag
	glslangValidator -V shaders/shader.frag -o shaders/frag.spv

.PHONY: test clean

test: computeConstant
	./computeConstant

clean:
	rm -f shaders/*.spv
	rm -f computeConstant
