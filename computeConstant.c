#include <vulkan/vulkan.h>

#include "computeConstant.h"


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
        if (CreateDebugUtilsMessengerEXT(state->instance, &createInfo, NULL, &callback) != VK_SUCCESS) {
            RUNTIME_ERROR("failed to set up debug callback!");
        }
        return callback;
}

QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
    QueueFamilyIndices indices = mkQueueFamilyIndices();

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, NULL);

    VkQueueFamilyProperties * queueFamilies = (VkQueueFamilyProperties *)
        malloc(queueFamilyCount * sizeof(VkQueueFamilyProperties));
    
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies);

    for (uint32_t i = 0; i < queueFamilyCount; i++) {
        VkQueueFamilyProperties queueFamily = queueFamilies[i];
        if (queueFamily.queueCount > 0 && queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT) {
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

    const VkPhysicalDeviceFeatures deviceFeatures = {};

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

VkDeviceMemory allocateMemory(VkAppState state){
    // Copied from https://gist.github.com/sheredom/523f02bbad2ae397d7ed255f3f3b5a7f

    VkPhysicalDeviceMemoryProperties properties;

    vkGetPhysicalDeviceMemoryProperties(state->physicalDevice, &properties);

    const int32_t bufferLength = 16384;

    const uint32_t bufferSize = sizeof(int32_t) * bufferLength;

    // we are going to need two buffers from this one memory
    const VkDeviceSize memorySize = bufferSize * 2; 

    // set memoryTypeIndex to an invalid entry in the properties.memoryTypes array
    uint32_t memoryTypeIndex = VK_MAX_MEMORY_TYPES;

    for (uint32_t k = 0; k < properties.memoryTypeCount; k++) {
      if ((VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT & properties.memoryTypes[k].propertyFlags) &&
        (VK_MEMORY_PROPERTY_HOST_COHERENT_BIT & properties.memoryTypes[k].propertyFlags) &&
        (memorySize < properties.memoryHeaps[properties.memoryTypes[k].heapIndex].size)) {
        memoryTypeIndex = k;
        break;
      }
    }

    if (memoryTypeIndex == VK_MAX_MEMORY_TYPES){
        RUNTIME_ERROR("Out of host memory error!");
    }

    const VkMemoryAllocateInfo memoryAllocateInfo = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = 0,
      .allocationSize = memorySize,
      .memoryTypeIndex = memoryTypeIndex
    };

    VkDeviceMemory memory;
    if(vkAllocateMemory(state->device, &memoryAllocateInfo, 0, &memory) != VK_SUCCESS){
        RUNTIME_ERROR("Cannot allocate memory");
    }
    
    return memory;
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

    return state;
}

void cleanup (VkAppState state) {
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
    cleanExtensionList(extensions);

    cleanup(state);
}