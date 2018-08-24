#include <vulkan/vulkan.h>

#include "vulkan_defns.h"
#include "fileContents.h"
#include "utils.h"

VkInstance createInstance(ExtensionInfo* extensions){

    const VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pNext = NULL,
        .pApplicationName = COMPUTE_CONSTANT_NAME,
        .applicationVersion = VK_MAKE_VERSION(0,0,0),
        .pEngineName = COMPUTE_CONSTANT_SHORT_NAME,
        .engineVersion = VK_MAKE_VERSION(0,0,0),
        .apiVersion = VK_API_VERSION_1_0,
    };

    uint32_t enabledLayerCount = 0;
    const char * const * ppEnabledLayerNames = NULL;
    if (enableValidationLayers){
        enabledLayerCount = SIZE(validationLayers);
        ppEnabledLayerNames = (const char * const *)validationLayers;    
    }

    const VkInstanceCreateInfo createInfo = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext = NULL,
        .pApplicationInfo = &app,
        .enabledLayerCount = enabledLayerCount,
        .ppEnabledLayerNames = ppEnabledLayerNames,
        .enabledExtensionCount = extensions->count,
        .ppEnabledExtensionNames = extensions->names
    };

    VkInstance instance;
    if (vkCreateInstance(&createInfo, NULL, &instance) != VK_SUCCESS) {
            RUNTIME_ERROR("failed to create instance!");
    }

    return instance;
}

VkDebugUtilsMessengerEXT setupDebugCallback(VkAppState state) {
        if (!enableValidationLayers) return NULL;

        VkInstance instance = state->instance;
        VkDebugUtilsMessengerEXT callback;
        const VkDebugUtilsMessengerCreateInfoEXT createInfo = {
            .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            .pNext = NULL,
            .messageSeverity =
                VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
            .messageType =
                VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
            .pfnUserCallback = &debugCallback
        };
        if (CreateDebugUtilsMessengerEXT(state->instance, &createInfo,
            NULL, &callback) != VK_SUCCESS) {
            RUNTIME_ERROR("failed to set up debug callback!");
        }
        return callback;
}

QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
    QueueFamilyIndices indices = mkQueueFamilyIndices();

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device,
        &queueFamilyCount, NULL);

    VkQueueFamilyProperties * queueFamilies = (VkQueueFamilyProperties *)
        malloc(queueFamilyCount * sizeof(VkQueueFamilyProperties));
    
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount,
        queueFamilies);

    for (uint32_t i = 0; i < queueFamilyCount; i++) {
        VkQueueFamilyProperties queueFamily = queueFamilies[i];
        if (queueFamily.queueCount > 0 &&
            queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT) {
            indices->computeFamily = i;
        }
    
        if (isComplete(indices)) {
            break;
        }
    }
    free(queueFamilies);
    return indices;
}

uint32_t isDiscreteGPU(VkPhysicalDevice gpu){
    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(gpu, &deviceProperties);

    return deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
}

int32_t rankDevice (VkPhysicalDevice gpu){
    QueueFamilyIndices indices = findQueueFamilies(gpu);

    uint8_t extensionsSupported = checkDeviceExtensionSupport(gpu);

    if (!(isComplete(indices) && extensionsSupported)){
        destroyQueueFamilyIndices(indices);
        return -1;
    }

    destroyQueueFamilyIndices(indices);
    return 1 + isDiscreteGPU(gpu);
}

VkPhysicalDevice pickPhysicalDevice(VkAppState state) {
    VkInstance instance = state->instance;

    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, NULL);

    if (deviceCount == 0) {
        RUNTIME_ERROR("failed to find GPUs with Vulkan support!");
    }

    VkPhysicalDevice * devices = (VkPhysicalDevice *)
        malloc(deviceCount * sizeof(VkPhysicalDevice));
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices);

    int32_t maxScore = -1;
    int32_t maxIdx = -1;
    for (uint32_t i = 0; i < deviceCount; i++){
        int32_t score = rankDevice(devices[i]);
        if (maxScore < score) {
            maxScore = score;
            maxIdx = i;
        }
    }
    if (maxIdx < 0) {
        RUNTIME_ERROR("failed to find a suitable GPU!");
    }
    VkPhysicalDevice device = devices[maxIdx];
    if (device == VK_NULL_HANDLE){
        RUNTIME_ERROR("failed to find a suitable GPU!");    
    }
    free(devices);
    return device;
}

VkDevice createLogicalDevice(VkAppState state) {
    VkPhysicalDevice physicalDevice = state->physicalDevice;
    QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

    int uniqueQueueFamilies[] = {indices->computeFamily};

    VkDeviceQueueCreateInfo * queueCreateInfos = (VkDeviceQueueCreateInfo *)
        malloc(sizeof(VkDeviceQueueCreateInfo) * SIZE(uniqueQueueFamilies));

    float queuePriority = 1.0f;
    for (uint32_t i = 0; i < SIZE(uniqueQueueFamilies); i++){
        const VkDeviceQueueCreateInfo queueCreateInfo = {
            .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .queueFamilyIndex = uniqueQueueFamilies[i],
            .queueCount = 1,
            .pQueuePriorities = &queuePriority
        };
        queueCreateInfos[i] = queueCreateInfo;
    }

    const VkPhysicalDeviceFeatures deviceFeatures = {
        .robustBufferAccess = VK_FALSE,
        .fullDrawIndexUint32 = VK_FALSE,
        .imageCubeArray = VK_FALSE,
        .independentBlend = VK_FALSE,
        .geometryShader = VK_FALSE,
        .tessellationShader = VK_FALSE,
        .sampleRateShading = VK_FALSE,
        .dualSrcBlend = VK_FALSE,
        .logicOp = VK_FALSE,
        .multiDrawIndirect = VK_FALSE,
        .drawIndirectFirstInstance = VK_FALSE,
        .depthClamp = VK_FALSE,
        .depthBiasClamp = VK_FALSE,
        .fillModeNonSolid = VK_FALSE,
        .depthBounds = VK_FALSE,
        .wideLines = VK_FALSE,
        .largePoints = VK_FALSE,
        .alphaToOne = VK_FALSE,
        .multiViewport = VK_FALSE,
        .samplerAnisotropy = VK_FALSE,
        .textureCompressionETC2 = VK_FALSE,
        .textureCompressionASTC_LDR = VK_FALSE,
        .textureCompressionBC = VK_FALSE,
        .occlusionQueryPrecise = VK_FALSE,
        .pipelineStatisticsQuery = VK_FALSE,
        .vertexPipelineStoresAndAtomics = VK_FALSE,
        .fragmentStoresAndAtomics = VK_FALSE,
        .shaderTessellationAndGeometryPointSize = VK_FALSE,
        .shaderImageGatherExtended = VK_FALSE,
        .shaderStorageImageExtendedFormats = VK_FALSE,
        .shaderStorageImageMultisample = VK_FALSE,
        .shaderStorageImageReadWithoutFormat = VK_FALSE,
        .shaderStorageImageWriteWithoutFormat = VK_FALSE,
        .shaderUniformBufferArrayDynamicIndexing = VK_FALSE,
        .shaderSampledImageArrayDynamicIndexing = VK_FALSE,
        .shaderStorageBufferArrayDynamicIndexing = VK_FALSE,
        .shaderStorageImageArrayDynamicIndexing = VK_FALSE,
        .shaderClipDistance = VK_FALSE,
        .shaderCullDistance = VK_FALSE,
        .shaderFloat64 = VK_FALSE,
        .shaderInt64 = VK_FALSE,
        .shaderInt16 = VK_FALSE,
        .shaderResourceResidency = VK_FALSE,
        .shaderResourceMinLod = VK_FALSE,
        .sparseBinding = VK_FALSE,
        .sparseResidencyBuffer = VK_FALSE,
        .sparseResidencyImage2D = VK_FALSE,
        .sparseResidencyImage3D = VK_FALSE,
        .sparseResidency2Samples = VK_FALSE,
        .sparseResidency4Samples = VK_FALSE,
        .sparseResidency8Samples = VK_FALSE,
        .sparseResidency16Samples = VK_FALSE,
        .sparseResidencyAliased = VK_FALSE,
        .variableMultisampleRate = VK_FALSE,
        .inheritedQueries = VK_FALSE,
    };

    uint32_t enabledLayerCount = 0;
    const char * const * ppEnabledLayerNames = NULL;
    if (enableValidationLayers){
        enabledLayerCount = SIZE(validationLayers);
        ppEnabledLayerNames = (const char * const *)validationLayers;
    }

    const VkDeviceCreateInfo createInfo = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = NULL,
        .queueCreateInfoCount = (uint32_t)(SIZE(uniqueQueueFamilies)),
        .pQueueCreateInfos = queueCreateInfos,
        .pEnabledFeatures = &deviceFeatures,
        .enabledExtensionCount = (uint32_t)(SIZE(deviceExtensions)),
        .ppEnabledExtensionNames = deviceExtensions,
        .enabledLayerCount = enabledLayerCount,
        .ppEnabledLayerNames = ppEnabledLayerNames
    };


    VkDevice device;
    if (vkCreateDevice(physicalDevice, &createInfo, NULL, &device) != VK_SUCCESS) {
        RUNTIME_ERROR("failed to create logical device!");
    }
    free(queueCreateInfos);
    destroyQueueFamilyIndices(indices);
    return device;
}

VkQueue getDeviceQueue(VkAppState state){
    QueueFamilyIndices indices = findQueueFamilies(state->physicalDevice);

    VkQueue computeQueue;
    vkGetDeviceQueue(state->device, indices->computeFamily, 0, &computeQueue);
    destroyQueueFamilyIndices(indices);
    return computeQueue;
}

uint32_t checkAll(uint32_t num_flags, const VkMemoryPropertyFlags flagList[],
    VkMemoryPropertyFlags flags){
    uint32_t res = 1;
    for (uint32_t k = 0; k < num_flags; k++){
        res = res && (flagList[k] & flags); 
    }
    return res;
}

uint32_t findMemoryType(VkPhysicalDeviceMemoryProperties properties,
    VkDeviceSize size, uint32_t typeFilter,
    uint32_t num_flags, const VkMemoryPropertyFlags flags[]){
    
    uint32_t memoryTypeIndex = VK_MAX_MEMORY_TYPES;

    for (uint32_t k = 0; k < properties.memoryTypeCount; k++) {
        VkMemoryType memoryType = properties.memoryTypes[k];
        if (checkAll(num_flags, flags, memoryType.propertyFlags) &&
            (size < properties.memoryHeaps[memoryType.heapIndex].size)) {

            memoryTypeIndex = k;
            break;
        }
    }
    if (memoryTypeIndex == VK_MAX_MEMORY_TYPES){
        RUNTIME_ERROR("Out of host memory error!");
    }
    return memoryTypeIndex;
}

VkDeviceMemory allocateMemory(VkAppState state){
    // Adapted from https://gist.github.com/sheredom/523f02bbad2ae397d7ed255f3f3b5a7f
    VkDevice device = state->device;
    VkBuffer buffer = state->computeBuffer;

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(device, buffer, &memReq);

    VkPhysicalDeviceMemoryProperties properties;
    vkGetPhysicalDeviceMemoryProperties(state->physicalDevice, &properties);

    const VkMemoryPropertyFlagBits flags[] = {
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    };
    uint32_t memoryTypeIndex = findMemoryType(properties, memReq.size,
        memReq.memoryTypeBits, SIZE(flags), flags);

    const VkMemoryAllocateInfo memoryAllocateInfo = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = 0,
      .allocationSize = memReq.size,
      .memoryTypeIndex = memoryTypeIndex
    };

    VkDeviceMemory memory;
    if(vkAllocateMemory(state->device, &memoryAllocateInfo, 0, &memory) != VK_SUCCESS){
        RUNTIME_ERROR("Cannot allocate memory");
    }

    if (vkBindBufferMemory(device, buffer, memory, 0)
        != VK_SUCCESS){
        RUNTIME_ERROR("Cannot bind memory to buffer");
    }
    
    return memory;
}

// We create a single buffer, and we will manage the offset later.
// ToRead: https://developer.nvidia.com/vulkan-memory-management
VkBuffer createBuffer(VkAppState state){
    VkDevice device = state->device;

    QueueFamilyIndices indices = findQueueFamilies(state->physicalDevice);
    int indiceList[] = {indices->computeFamily};
    destroyQueueFamilyIndices(indices);

    const VkBufferCreateInfo bufferCreateInfo = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .pNext = NULL,
      .flags = 0,
      .size  = state->deviceMemorySize,
      .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .queueFamilyIndexCount = 1,
      .pQueueFamilyIndices = indiceList
    };

    VkBuffer buffer;
    if (vkCreateBuffer(device, &bufferCreateInfo, 0, &buffer)
        != VK_SUCCESS){
        RUNTIME_ERROR("Cannot create buffer");
    }

    return buffer;
}

void fillMemory(VkAppState state, int32_t *data){
    VkDevice device = state->device;
    VkDeviceMemory memory = state->deviceMemory;
    VkDeviceSize memorySize = state->deviceMemorySize;

    int32_t *payload;
    if (vkMapMemory(device, memory, 0, memorySize, 0, (void *)&payload)
        != VK_SUCCESS){
        RUNTIME_ERROR("Cannot map device memory");
    }

    if (memcpy(payload, data, memorySize)
         == NULL){
        RUNTIME_ERROR("Cannot copy data to GPU memory");
    }
    vkUnmapMemory(device, memory);
}

VkShaderModule createShaderModule(VkAppState state, FileContents shader) {
    VkDevice device = state->device;
    VkShaderModuleCreateInfo createInfo = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .codeSize = shader->size,
        .pCode = (const uint32_t*)shader->code,
    };

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &createInfo, NULL, &shaderModule)
        != VK_SUCCESS) {

        RUNTIME_ERROR("failed to create shader module!");
    }
    return shaderModule;
}

VkDescriptorSetLayout createDescriptorSetLayout(VkAppState state){
    VkDevice device = state->device;
    // XXX: hardcoded two descriptor set layouts (i.e. two "locations" in shader)
    VkDescriptorSetLayoutBinding descriptorSetLayoutBindings[2] = {
      {
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .pImmutableSamplers = NULL
      },
       {
        .binding = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .pImmutableSamplers = NULL
      }
    };

    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .pNext = NULL,
      .flags = 0,
      .bindingCount = 2,
      .pBindings = descriptorSetLayoutBindings
    };

    VkDescriptorSetLayout descriptorSetLayout;
    if (vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo,
        0, &descriptorSetLayout) != VK_SUCCESS){
        RUNTIME_ERROR("Cannot create descriptor set layout");
    }
    
    return descriptorSetLayout;
}

VkPipelineLayout createPipelineLayout(VkAppState state){
    VkDevice device = state->device;
    VkDescriptorSetLayout descriptorSetLayout = state->descriptorSetLayout;
    
    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pNext = NULL,
      .flags = 0,
      .setLayoutCount = 1,
      .pSetLayouts = &descriptorSetLayout,
      .pushConstantRangeCount = 0,
      .pPushConstantRanges = NULL
    };

    VkPipelineLayout pipelineLayout;
    if (vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, 0, &pipelineLayout)
        != VK_SUCCESS){
        RUNTIME_ERROR("Cannot create pipeline layout");
    }

    return pipelineLayout;
}

VkPipeline createComputePipeline(VkAppState state){
    VkDevice device = state->device;
    VkPipelineLayout pipelineLayout = state->pipelineLayout;
    // XXX: Shader hardcoded
    FileContents shader = readFile("../shaders/reduceShader.comp.spv");

    VkShaderModule module = createShaderModule(state, shader);

    VkComputePipelineCreateInfo computePipelineCreateInfo = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .pNext = NULL,
      .flags = 0,
      .stage = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = module,
        .pName = "main", // XXX: Hardcoded name
        .pSpecializationInfo = NULL
      },
      .layout = pipelineLayout,
      .basePipelineHandle = VK_NULL_HANDLE,
      .basePipelineIndex = 0
    };

    VkPipeline pipeline;
    if (vkCreateComputePipelines(device, 0, 1, &computePipelineCreateInfo,
        0, &pipeline) != VK_SUCCESS){
        RUNTIME_ERROR("Cannot create compute pipeline");
    }
    
    destroyFileContents(shader);
    vkDestroyShaderModule(device, module, NULL);
    return pipeline;
}

VkDescriptorPool createDescriptorPool(VkAppState state){
    VkDevice device = state->device;

    VkDescriptorPoolSize descriptorPoolSize = {
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 4
    };

    VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .pNext = NULL,
      .flags = 0,
      .maxSets = 2,
      .poolSizeCount = 1,
      .pPoolSizes = &descriptorPoolSize
    };

    VkDescriptorPool descriptorPool;
    if (vkCreateDescriptorPool(device, &descriptorPoolCreateInfo,
        0, &descriptorPool) != VK_SUCCESS){
        RUNTIME_ERROR("Cannot create descriptor pool");
    }
    return descriptorPool;
}

VkDescriptorSet * createDescriptorSet(VkAppState state){
    VkDevice device = state->device;
    VkDescriptorSetLayout descriptorSetLayout = state->descriptorSetLayout;
    VkDescriptorPool descriptorPool = state->descriptorPool;
    VkBuffer buffer = state->computeBuffer;


    VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .pNext = NULL,
        .descriptorPool = descriptorPool,
        .descriptorSetCount = 1,
        .pSetLayouts = &descriptorSetLayout
    };

    // We allocate two descriptor sets with input/output descriptor
    // sets switched (ping-pong input/output buffers)
    VkDescriptorSet * descriptorSet =
        (VkDescriptorSet *)malloc(2 * sizeof(VkDescriptorSet));
    if (vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo,
        &descriptorSet[0]) != VK_SUCCESS){
        RUNTIME_ERROR("Cannot allocate descriptor sets");
    }
    VkDescriptorSet descriptorSet_out;
    if (vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo,
        &descriptorSet[1]) != VK_SUCCESS){
        RUNTIME_ERROR("Cannot allocate descriptor sets");
    }
    
    // XXX: hardcoded offsets and ranges: two layouts, input and output
    VkDescriptorBufferInfo in_descriptorBufferInfo = {
      .buffer = buffer,
      .offset = 0,
      .range = state->deviceMemorySize/2
    };

    VkDescriptorBufferInfo out_descriptorBufferInfo = {
      .buffer = buffer,
      .offset = state->deviceMemorySize/2,
      .range = VK_WHOLE_SIZE
    };

    VkWriteDescriptorSet writeDescriptorSet[4] = {
      {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext = NULL,
        .dstSet = descriptorSet[0],
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pImageInfo = NULL,
        .pBufferInfo = &in_descriptorBufferInfo,
        .pTexelBufferView = NULL
      },
      {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext = NULL,
        .dstSet = descriptorSet[0],
        .dstBinding = 1,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pImageInfo = NULL,
        .pBufferInfo = &out_descriptorBufferInfo,
        .pTexelBufferView = NULL
      },
      {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext = NULL,
        .dstSet = descriptorSet[1],
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pImageInfo = NULL,
        .pBufferInfo = &out_descriptorBufferInfo,
        .pTexelBufferView = NULL
      },
      {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext = NULL,
        .dstSet = descriptorSet[1],
        .dstBinding = 1,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pImageInfo = NULL,
        .pBufferInfo = &in_descriptorBufferInfo,
        .pTexelBufferView = NULL
      },
    };

    vkUpdateDescriptorSets(device, 4, writeDescriptorSet, 0, 0);
    return descriptorSet;
}

VkCommandPool createCommandPool(VkAppState state){
    VkDevice device = state->device;

    // XXX: keep index somewhere
    QueueFamilyIndices indices = findQueueFamilies(state->physicalDevice);
    int queueFamilyIndex = indices->computeFamily;
    destroyQueueFamilyIndices(indices);

    VkCommandPoolCreateInfo commandPoolCreateInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .queueFamilyIndex = queueFamilyIndex
    };

    VkCommandPool commandPool;
    if (vkCreateCommandPool(device, &commandPoolCreateInfo, 0, &commandPool)
        != VK_SUCCESS){
        RUNTIME_ERROR("Cannot create command pool");
    }

    return commandPool;
}

VkCommandBuffer createCommandBuffer(VkAppState state){
    VkCommandPool commandPool = state->commandPool;

    VkCommandBufferAllocateInfo commandBufferAllocateInfo = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .pNext = NULL,
      .commandPool = commandPool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1
    };

    VkDevice device = state->device;

    VkCommandBuffer commandBuffer;
    if (vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &commandBuffer)
        != VK_SUCCESS) {
        RUNTIME_ERROR("Cannot allocate command buffer");
    }

    VkCommandBufferBeginInfo commandBufferBeginInfo = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .pNext = NULL,
      .flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT,
      .pInheritanceInfo = NULL
    };

    if (vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo)
        != VK_SUCCESS){
        RUNTIME_ERROR("Cannot begin command buffer");
    }

    VkPipeline pipeline = state->computePipeline;

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

    VkPipelineLayout pipelineLayout = state->pipelineLayout;
    VkDescriptorSet * descriptorSet = state->descriptorSet;

    VkDeviceSize bufferSize = state->deviceMemorySize / 2;
    uint32_t num_ws = bufferSize / sizeof(int32_t);
    uint32_t turn = 1;
    while (num_ws > 1) { 
        turn = 1 - turn;

        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
            pipelineLayout, 0, 1, &descriptorSet[turn], 0, 0);

        vkCmdDispatch(commandBuffer, num_ws, 1, 1);
        num_ws = num_ws / 2;
    }

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS){
        RUNTIME_ERROR("Cannot end command buffer");
    }
    return commandBuffer;
}

VkAppState initVulkan(ExtensionInfo * requiredExtensions){
    VkAppState state = (VkAppState)malloc(sizeof(VkAppState_T));
    
    VkInstance instance = createInstance(requiredExtensions);
    state->instance = instance;
    
    VkDebugUtilsMessengerEXT callback = setupDebugCallback(state);
    state->callback = callback;
    
    VkPhysicalDevice pdevice = pickPhysicalDevice(state);
    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(pdevice, &deviceProperties);
    fprintf(stdout, "Selected physical device: %s\n", deviceProperties.deviceName);
    state->physicalDevice = pdevice;

    VkDevice device = createLogicalDevice(state);
    state->device = device;

    VkQueue queue = getDeviceQueue(state);
    state->computeQueue = queue;

    // XXX: Hardcoded buffer length + memory size
    const int32_t bufferLength = 16384;
    const uint32_t bufferSize = sizeof(int32_t) * bufferLength;
    const VkDeviceSize memorySize = bufferSize * 2;
    state->deviceMemorySize = memorySize;

    VkBuffer computeBuffer = createBuffer(state);
    state->computeBuffer = computeBuffer;

    VkDeviceMemory deviceMemory = allocateMemory(state);
    state->deviceMemory = deviceMemory;
    
    // XXX: hardcoded fill device memory with stuff
    int32_t *payload = (int32_t *) malloc(bufferLength * 2 * sizeof(int32_t));
    int32_t expected = 0;
    for (int32_t i = 0; i < bufferLength; i++){
        payload[i] = 42;
        expected += payload[i];
    }
    fprintf(stdout, "Expecting result %d\n", expected);
    for (int32_t i = bufferLength; i < bufferLength*2; i++){
        payload[i] = 100;
    }
    fillMemory(state, payload);
    free(payload);

    VkDescriptorSetLayout descriptorSetLayout =
        createDescriptorSetLayout(state);
    state->descriptorSetLayout = descriptorSetLayout;

    VkPipelineLayout pipelineLayout =
        createPipelineLayout(state);
    state->pipelineLayout = pipelineLayout;

    VkPipeline computePipeline = createComputePipeline(state);
    state->computePipeline = computePipeline;

    VkDescriptorPool descriptorPool = createDescriptorPool(state);
    state->descriptorPool = descriptorPool;

    VkDescriptorSet * descriptorSet = createDescriptorSet(state);
    state->descriptorSet = descriptorSet;

    VkCommandPool commandPool = createCommandPool(state);
    state->commandPool = commandPool;

    VkCommandBuffer commandBuffer = createCommandBuffer(state);
    state->commandBuffer = commandBuffer;

    return state;
}

void cleanup (VkAppState state) {
    vkFreeCommandBuffers(state->device, state->commandPool, 1, &state->commandBuffer);
    vkDestroyCommandPool(state->device, state->commandPool, NULL);
    vkDestroyDescriptorPool(state->device, state->descriptorPool, NULL);
    vkDestroyDescriptorSetLayout(state->device,
        state->descriptorSetLayout, NULL);
    free(state->descriptorSet);
    vkDestroyPipelineLayout(state->device, state->pipelineLayout, NULL);
    vkDestroyPipeline(state->device, state->computePipeline, NULL);
    vkDestroyBuffer(state->device, state->computeBuffer, NULL);
    vkFreeMemory(state->device, state->deviceMemory, NULL);

    if (enableValidationLayers){
        DestroyDebugUtilsMessengerEXT(state->instance, state->callback, NULL);
    }
    vkDestroyDevice(state->device, NULL);
    vkDestroyInstance(state->instance, NULL);
    free(state);
}

int main(int main, char **argv){
    ExtensionInfo * extensions = getRequiredExtensions();
    PRINT_LIST(stdout, "Requesting extensions: ", extensions->count, extensions->names);

    VkAppState state = initVulkan(extensions);

    // We now have a queue in state, and everything set up to submit a command to the queue

    VkSubmitInfo submitInfo = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = NULL, 
        .waitSemaphoreCount = 0,
        .pWaitSemaphores = NULL,
        .pWaitDstStageMask = NULL,
        .commandBufferCount = 1,
        .pCommandBuffers = &state->commandBuffer,
        .signalSemaphoreCount = 0,
        .pSignalSemaphores = NULL
    };
    
    uint32_t num_ws = state->deviceMemorySize / (2 * (sizeof(uint32_t)));
    uint32_t mem_size = state->deviceMemorySize / 2;

    int32_t * payload;
    if (vkMapMemory(state->device, state->deviceMemory, 0,
        state->deviceMemorySize, 0, (void *)&payload) != VK_SUCCESS){
        RUNTIME_ERROR("Error mapping memory");
    }

    if (vkQueueSubmit(state->computeQueue, 1, &submitInfo, 0) != VK_SUCCESS){
        RUNTIME_ERROR("Cannot submit to compute queue");
    }
    if (vkQueueWaitIdle(state->computeQueue) != VK_SUCCESS){
        RUNTIME_ERROR("Error waiting for computeQueue");
    }

    fprintf(stdout, "result = %d\n", payload[0]);
    vkUnmapMemory(state->device, state->deviceMemory);

    // End computation

    cleanExtensionList(extensions);

    cleanup(state);
}