VULKAN_SDK_PATH = /home/dcastrop/Vulkan/1.1.82.0/x86_64
INCLUDE_DIR = ../include
SHADER_DIR = ../shaders
VALGRIND_CFG = ../valgrind
CFLAGS = -I$(VULKAN_SDK_PATH)/include -I$(INCLUDE_DIR)
LDFLAGS = -L$(VULKAN_SDK_PATH)/lib `pkg-config --static --libs glfw3` -lvulkan

reduce: reduce.c $(SHADER_DIR)/reduceShader.comp.spv
	gcc $(CFLAGS) -o reduce reduce.c $(LDFLAGS)

%.spv: %.glsl
	glslangValidator -V $< -o $@

nodebug: reduce.c $(SHADER_DIR)/reduceShader.comp.spv
	gcc $(CFLAGS) -o reduce -DNDEBUG reduce.c $(LDFLAGS)

.PHONY: test clean valgrind

valgrind: reduce $(VALGRIND_CFG)/valgrind.supp
	valgrind --suppressions=$(VALGRIND_CFG)/valgrind.supp --leak-check=full ./reduce > /dev/null

test: reduce
	./reduce

clean:
	rm -f shaders/*.spv
	rm -f reduce
