VULKAN_SDK_PATH = /home/dcastrop/Vulkan/1.1.82.0/x86_64
CFLAGS = -I$(VULKAN_SDK_PATH)/include
LDFLAGS = -L$(VULKAN_SDK_PATH)/lib `pkg-config --static --libs glfw3` -lvulkan

computeConstant: computeConstant.c computeConstant.h shaders/vert.spv shaders/frag.spv
	gcc $(CFLAGS) -o computeConstant computeConstant.c $(LDFLAGS)

shaders/vert.spv: shaders/shader.vert
	glslangValidator -V shaders/shader.vert -o shaders/vert.spv

shaders/frag.spv: shaders/shader.frag
	glslangValidator -V shaders/shader.frag -o shaders/frag.spv

nodebug: computeConstant.c computeConstant.h shaders/vert.spv shaders/frag.spv
	gcc $(CFLAGS) -o computeConstant -DNDEBUG computeConstant.c $(LDFLAGS)

old: computeConstant.cpp computeConstant.h shaders/vert.spv shaders/frag.spv
	g++ $(CFLAGS) -o computeConstant computeConstant.cpp $(LDFLAGS)

.PHONY: test clean

test: computeConstant
	./computeConstant

clean:
	rm -f shaders/*.spv
	rm -f computeConstant
