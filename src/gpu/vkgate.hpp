// gpu/vkgate.hpp: the phase gate (gate.comp) on the integrated GPU through Vulkan compute. One VkGate per
// process; run() dispatches INVOCATIONS x PER indices from BASE and returns the passing indices (unsorted).
// The output buffer is host-visible and coherent (an integrated GPU shares memory with the CPU).
#pragma once
#include <vulkan/vulkan.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

struct VkGate {
    VkInstance inst = VK_NULL_HANDLE; VkPhysicalDevice phys = VK_NULL_HANDLE; VkDevice dev = VK_NULL_HANDLE; VkQueue queue = VK_NULL_HANDLE;
    VkBuffer buf = VK_NULL_HANDLE; VkDeviceMemory mem = VK_NULL_HANDLE; void *map = nullptr; uint32_t cap = 0;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE; VkPipelineLayout pl = VK_NULL_HANDLE; VkPipeline pipe = VK_NULL_HANDLE; VkShaderModule sm = VK_NULL_HANDLE;
    VkDescriptorPool dp = VK_NULL_HANDLE; VkDescriptorSet ds = VK_NULL_HANDLE; VkCommandPool cp = VK_NULL_HANDLE; VkCommandBuffer cb = VK_NULL_HANDLE; VkFence fence = VK_NULL_HANDLE;
    std::string name;
    struct PC { uint64_t base, lim0, lim1, lim2; uint32_t per, pad; };
    static void ck(VkResult r, const char *what) { if (r != VK_SUCCESS) throw std::runtime_error(std::string(what) + " failed: " + std::to_string(r)); }

    void init(const std::string &spv, uint32_t capacity) {
        cap = capacity;
        VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO}; ai.apiVersion = VK_API_VERSION_1_2; ai.pApplicationName = "snowy-gate";
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo = &ai; ck(vkCreateInstance(&ici, nullptr, &inst), "vkCreateInstance");
        uint32_t n = 0; vkEnumeratePhysicalDevices(inst, &n, nullptr); std::vector<VkPhysicalDevice> pds(n); vkEnumeratePhysicalDevices(inst, &n, pds.data());
        for (auto p : pds) { VkPhysicalDeviceProperties pr; vkGetPhysicalDeviceProperties(p, &pr); VkPhysicalDeviceFeatures f; vkGetPhysicalDeviceFeatures(p, &f);
            if (pr.deviceType != VK_PHYSICAL_DEVICE_TYPE_CPU && f.shaderInt64) { phys = p; name = pr.deviceName; break; } }
        if (!phys) throw std::runtime_error("no GPU with shaderInt64");
        uint32_t qn = 0; vkGetPhysicalDeviceQueueFamilyProperties(phys, &qn, nullptr); std::vector<VkQueueFamilyProperties> qf(qn); vkGetPhysicalDeviceQueueFamilyProperties(phys, &qn, qf.data());
        uint32_t qfi = UINT32_MAX; for (uint32_t i = 0; i < qn; i++) if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfi = i; break; }
        float prio = 1.0f; VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; qci.queueFamilyIndex = qfi; qci.queueCount = 1; qci.pQueuePriorities = &prio;
        VkPhysicalDeviceFeatures feat{}; feat.shaderInt64 = VK_TRUE;
        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci; dci.pEnabledFeatures = &feat;
        ck(vkCreateDevice(phys, &dci, nullptr, &dev), "vkCreateDevice"); vkGetDeviceQueue(dev, qfi, 0, &queue);
        // output buffer: count, cap, then cap indices
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bci.size = 8 + 8ull * cap; bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ck(vkCreateBuffer(dev, &bci, nullptr, &buf), "vkCreateBuffer");
        VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, buf, &mr); VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(phys, &mp);
        uint32_t mt = UINT32_MAX; const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        for (uint32_t i = 0; i < mp.memoryTypeCount; i++) if ((mr.memoryTypeBits >> i & 1) && (mp.memoryTypes[i].propertyFlags & want) == want) { mt = i; break; }
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; mai.allocationSize = mr.size; mai.memoryTypeIndex = mt; ck(vkAllocateMemory(dev, &mai, nullptr, &mem), "vkAllocateMemory");
        vkBindBufferMemory(dev, buf, mem, 0); ck(vkMapMemory(dev, mem, 0, VK_WHOLE_SIZE, 0, &map), "vkMapMemory");
        // pipeline
        std::ifstream in(spv, std::ios::binary); std::vector<char> code((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>()); if (code.empty()) throw std::runtime_error("cannot read " + spv);
        VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; smci.codeSize = code.size(); smci.pCode = (const uint32_t *) code.data(); ck(vkCreateShaderModule(dev, &smci, nullptr, &sm), "vkCreateShaderModule");
        VkDescriptorSetLayoutBinding b{}; b.binding = 0; b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; b.descriptorCount = 1; b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        VkDescriptorSetLayoutCreateInfo dlci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; dlci.bindingCount = 1; dlci.pBindings = &b; ck(vkCreateDescriptorSetLayout(dev, &dlci, nullptr, &dsl), "dsl");
        VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PC)};
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; plci.setLayoutCount = 1; plci.pSetLayouts = &dsl; plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr; ck(vkCreatePipelineLayout(dev, &plci, nullptr, &pl), "pl");
        VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; cpci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}; cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cpci.stage.module = sm; cpci.stage.pName = "main"; cpci.layout = pl;
        ck(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipe), "pipeline");
        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1}; VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; dpci.maxSets = 1; dpci.poolSizeCount = 1; dpci.pPoolSizes = &ps; ck(vkCreateDescriptorPool(dev, &dpci, nullptr, &dp), "dp");
        VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; dsai.descriptorPool = dp; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &dsl; ck(vkAllocateDescriptorSets(dev, &dsai, &ds), "ds");
        VkDescriptorBufferInfo dbi{buf, 0, VK_WHOLE_SIZE}; VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; w.dstSet = ds; w.dstBinding = 0; w.descriptorCount = 1; w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w.pBufferInfo = &dbi;
        vkUpdateDescriptorSets(dev, 1, &w, 0, nullptr);
        VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; cpi.queueFamilyIndex = qfi; cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; ck(vkCreateCommandPool(dev, &cpi, nullptr, &cp), "cp");
        VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; cbai.commandPool = cp; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount = 1; ck(vkAllocateCommandBuffers(dev, &cbai, &cb), "cb");
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; ck(vkCreateFence(dev, &fci, nullptr, &fence), "fence");
    }
    // indices BASE .. BASE + groups*64*PER - 1; passing ones appended to OUT (returns false if the buffer overflowed)
    bool run(uint64_t base, uint32_t groups, uint32_t per, const uint64_t *lims, std::vector<uint64_t> &out) {
        uint32_t *hdr = (uint32_t *) map; hdr[0] = 0; hdr[1] = cap;
        vkResetCommandBuffer(cb, 0); VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT; vkBeginCommandBuffer(cb, &bi);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe); vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, nullptr);
        PC pc{base, lims[0], lims[1], lims[2], per, 0}; vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
        vkCmdDispatch(cb, groups, 1, 1);
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER}; mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
        vkEndCommandBuffer(cb);
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cb; vkResetFences(dev, 1, &fence); ck(vkQueueSubmit(queue, 1, &si, fence), "submit");
        ck(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX), "wait");
        const uint32_t cnt = hdr[0]; const uint64_t *idx = (const uint64_t *) ((const char *) map + 8);
        out.insert(out.end(), idx, idx + std::min(cnt, cap));
        return cnt <= cap;
    }
};
