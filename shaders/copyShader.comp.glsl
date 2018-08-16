#version 450
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0, std430) buffer in_b
{
    int b[16384];
} lay0;

layout(binding = 1, std430) buffer out_b
{
    int b[16384];
} lay1;

void main()
{
    lay1.b[gl_GlobalInvocationID.x] = 1 + lay0.b[gl_GlobalInvocationID.x];
}

