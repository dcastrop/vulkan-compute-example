# Experiments with Compute Shaders in Vulkan

This is vulkan project that runs a compute shader that effectively applies `reduce (+)` to an array of data.

Based on [http://www.duskborn.com/a-simple-vulkan-compute-example/](http://www.duskborn.com/a-simple-vulkan-compute-example/).
A small minor difference is that instead of creating two buffers, we create a
single one and use the offsets in VkDescriptorBufferInfo.

## To-do
- [ ] Allocate two kinds of memory: Local, and Host+Coherent
- [ ] Issue copy commands to transfer memory from the host to the local memory,
and back to retrieve the results.
- [ ] Generalise `reduce` to work on sequences of arbitrary size (ensure that
  alignment requirements are satisfied, etc.)
- [ ] Two versions: "in-place" reduce, and ping-pong input/output buffers.

## Issues (Fixed by ping-pong-ing input/output buffers):

- [x] Careless use of vkQueueSubmit: should batch log(N) commands, and submit once
- [x] Avoid the use of memcpy to set the input data for each iteration
