#version 450
layout(local_size_x = 1024) in;

layout(binding = 0, std430) buffer in_b
{
    uint b[16384];
} lay0;

layout(binding = 1, std430) buffer out_b
{
    uint b[16384];
} lay1;

void main()
{
    lay1.b[gl_GlobalInvocationID.x] = lay0.b[gl_GlobalInvocationID.x] + lay0.b[gl_GlobalInvocationID.x + gl_NumWorkGroups.x];
}

