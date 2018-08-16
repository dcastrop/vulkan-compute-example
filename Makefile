VULKAN_SDK_PATH = /home/dcastrop/Vulkan/1.1.82.0/x86_64
CFLAGS = -I$(VULKAN_SDK_PATH)/include
LDFLAGS = -L$(VULKAN_SDK_PATH)/lib `pkg-config --static --libs glfw3` -lvulkan

computeConstant: computeConstant.c computeConstant.h utils.h fileContents.h shaders/copyShader.spv
	gcc $(CFLAGS) -o computeConstant computeConstant.c $(LDFLAGS)

shaders/copyShader.spv: shaders/copyShader.comp
	glslangValidator -V shaders/copyShader.comp -o shaders/copyShader.spv

nodebug: computeConstant.c computeConstant.h shaders/vert.spv shaders/frag.spv
	gcc $(CFLAGS) -o computeConstant -DNDEBUG computeConstant.c $(LDFLAGS)

.PHONY: test clean

test: computeConstant
	./computeConstant

clean:
	rm -f shaders/*.spv
	rm -f computeConstant
