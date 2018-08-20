VULKAN_SDK_PATH = /home/dcastrop/Vulkan/1.1.82.0/x86_64
INCLUDE_DIR = ../include
SHADER_DIR = ../shaders
VALGRIND_CFG = ../valgrind
CFLAGS = -I$(VULKAN_SDK_PATH)/include -I$(INCLUDE_DIR)
LDFLAGS = -L$(VULKAN_SDK_PATH)/lib `pkg-config --static --libs glfw3` -lvulkan

pmap: pmap.c $(SHADER_DIR)/copyShader.comp.spv
	gcc $(CFLAGS) -o pmap pmap.c $(LDFLAGS)

%.spv: %.glsl
	glslangValidator -V $< -o $@

nodebug: pmap.c $(SHADER_DIR)/copyShader.comp.spv
	gcc $(CFLAGS) -o pmap -DNDEBUG pmap.c $(LDFLAGS)

.PHONY: test clean valgrind

valgrind: pmap $(VALGRIND_CFG)/valgrind.supp
	valgrind --suppressions=$(VALGRIND_CFG)/valgrind.supp --leak-check=full ./pmap > /dev/null

test: pmap
	./pmap

clean:
	rm -f shaders/*.spv
	rm -f pmap
