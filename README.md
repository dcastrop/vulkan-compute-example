# Experiments with Compute Shaders in Vulkan

This is a compute shader that returns a constant, based on
[http://www.duskborn.com/a-simple-vulkan-compute-example/](http://www.duskborn.com/a-simple-vulkan-compute-example/).

A small minor difference is that instead of creating two buffers, we create a
single one and use the offsets in VkDescriptorBufferInfo.

