VULKAN_SDK_PATH = /home/dcastrop/Vulkan/1.1.82.0/x86_64
INCLUDE_DIR = ./include
CFLAGS = -I$(VULKAN_SDK_PATH)/include -I$(INCLUDE_DIR)
LDFLAGS = -L$(VULKAN_SDK_PATH)/lib `pkg-config --static --libs glfw3` -lvulkan

computeConstant: computeConstant.c shaders/copyShader.comp.spv
	gcc $(CFLAGS) -o computeConstant computeConstant.c $(LDFLAGS)

%.spv: %.glsl
	glslangValidator -V $< -o $@

nodebug: computeConstant.c shaders/copyShader.comp.spv
	gcc $(CFLAGS) -o computeConstant -DNDEBUG computeConstant.c $(LDFLAGS)

.PHONY: test clean valgrind

valgrind: computeConstant valgrind/valgrind.supp
	valgrind --suppressions=valgrind/valgrind.supp --leak-check=full ./computeConstant > /dev/null

test: computeConstant
	./computeConstant

clean:
	rm -f shaders/*.spv
	rm -f computeConstant
