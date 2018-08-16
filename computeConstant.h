#ifndef __COMPUTE_CONSTANT__
#define __COMPUTE_CONSTANT__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "utils.h"

#ifdef NDEBUG
const uint8_t enableValidationLayers = 0;
#else
const uint8_t enableValidationLayers = 1;
#endif

#define COMPUTE_CONSTANT_NAME "ComputeConstant"
#define COMPUTE_CONSTANT_SHORT_NAME "const"

#define NO_VALIDATION_LAYERS -1

const char * validationLayers[] = {
    "VK_LAYER_LUNARG_standard_validation"
};

const char * deviceExtensions[] = {
};

uint8_t checkValidationLayerSupport() {
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, NULL);

    VkLayerProperties * availableLayers =
        (VkLayerProperties *) malloc(layerCount * (sizeof(VkLayerProperties)));
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers);

    for (uint32_t i = 0; i < SIZE(validationLayers); i++) {
        uint8_t layerFound = 0;

        for (uint32_t j = 0; j < layerCount; j++) {
            if (strcmp(validationLayers[i], availableLayers[j].layerName) == 0) {
                    layerFound = 1;
                    break;
                }
            }

            if (!layerFound) {
                free(availableLayers);
                return 0;
            }
    }

    free(availableLayers);
    return 1;
}

void checkValidationExtension(){
    if (enableValidationLayers && !checkValidationLayerSupport()) {
        RUNTIME_ERROR("Validation layers required but not available");
    }
}

typedef struct extInfo {
    uint32_t count;
    const char * const * names;  
} ExtensionInfo;

ExtensionInfo* getRequiredExtensions(){
    
    uint32_t numExts = 0;
    const char **extNames = NULL;

    if (enableValidationLayers) {
        checkValidationExtension();
        extNames = (const char **)
            realloc(extNames, (numExts + 1) * sizeof(const char *));    
        extNames[numExts++] = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
    }

    ExtensionInfo * extensions =
        (ExtensionInfo *) malloc(sizeof(ExtensionInfo));
    
    extensions->count = numExts;
    extensions->names = (const char * const *) extNames;
    return extensions;
}

void cleanExtensionList(ExtensionInfo *extensions){
    if (extensions->count > 0) {
        free((const char **)extensions->names);
    }
    free(extensions);
}

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance,
    const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkDebugUtilsMessengerEXT* pCallback) {

    PFN_vkCreateDebugUtilsMessengerEXT func =
        (PFN_vkCreateDebugUtilsMessengerEXT)
        vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != NULL) {
        return (*func)(instance, pCreateInfo, pAllocator, pCallback);
    } else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance,
    VkDebugUtilsMessengerEXT callback,
    const VkAllocationCallbacks* pAllocator) {

    PFN_vkDestroyDebugUtilsMessengerEXT func =
        (PFN_vkDestroyDebugUtilsMessengerEXT)
        vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != NULL) {
        (*func)(instance, callback, pAllocator);
    }
}

VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData) {

        if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT) {
            fprintf(stderr, "VERBOSE : ");
        } else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) {
            fprintf(stderr, "INFO : ");
        } else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
            fprintf(stderr, "WARNING : ");
        } else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
            fprintf(stderr, "ERROR : ");
        }

        fprintf(stderr, pCallbackData->pMessage);
        fprintf(stderr, "\n");
        return VK_FALSE;
    }

typedef struct qfIndices {
    int computeFamily;
} QueueFamilyIndices_T;

typedef QueueFamilyIndices_T * QueueFamilyIndices;

QueueFamilyIndices mkQueueFamilyIndices(){
    QueueFamilyIndices qIdx = (QueueFamilyIndices) malloc(sizeof(QueueFamilyIndices_T));
    qIdx->computeFamily = -1;
    return qIdx;
}

void destroyQueueFamilyIndices(QueueFamilyIndices idx){
    if(idx){
        free(idx);
    }
}

uint8_t isComplete(QueueFamilyIndices idx){
    return (idx->computeFamily >= 0);
}

uint8_t checkDeviceExtensionSupport(VkPhysicalDevice device) {
    uint32_t extensionCount;
    vkEnumerateDeviceExtensionProperties(device, NULL, &extensionCount, NULL);

    VkExtensionProperties * availableExtensions = (VkExtensionProperties *)
        malloc(sizeof(VkExtensionProperties) * extensionCount);
    vkEnumerateDeviceExtensionProperties(device, NULL, &extensionCount, availableExtensions);

    for (uint32_t i = 0; i < SIZE(deviceExtensions); i++){
        uint8_t found = 0;
        for (uint32_t j = 0; j < extensionCount; j++) {
            if (!strcmp(deviceExtensions[i], availableExtensions[j].extensionName)){
                found = 1;
                break;
            }
        }
        if (!found){
            free(availableExtensions);
            return 0;
        }
    }
    free(availableExtensions);
    return 1;
}

typedef struct VAppState {
    VkInstance instance;
    VkDebugUtilsMessengerEXT callback;
    VkPhysicalDevice physicalDevice;
    VkDevice device;
    VkQueue computeQueue;
    VkDeviceSize deviceMemorySize;
    VkDeviceMemory deviceMemory;
    VkBuffer computeBuffer;
    VkDescriptorSetLayout descriptorSetLayout;
    VkPipelineLayout pipelineLayout;
    VkPipeline computePipeline;
    VkDescriptorPool descriptorPool;
    VkDescriptorSet descriptorSet;
    VkCommandPool commandPool;
    VkCommandBuffer commandBuffer;
} VkAppState_T;

typedef VkAppState_T* VkAppState;

#endif