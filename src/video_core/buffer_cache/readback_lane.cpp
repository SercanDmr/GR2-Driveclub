// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/logging/log.h"
#include "common/polyfill_thread.h"
#include "common/thread.h"
#include "core/memory.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/buffer_cache/readback_lane.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_platform.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_semaphore.h"

namespace VideoCore {

struct ReadbackLanes::Lane {
    Lane(const Vulkan::Instance& instance, vk::Queue queue_, u32 family, bool shared_queue_,
         u64 staging_size)
        : queue{queue_}, shared_queue{shared_queue_}, timeline{instance},
          staging{instance, 0, staging_size, MemoryType::HostCached, "Readback lane"} {
        const vk::Device device = instance.GetDevice();
        const vk::CommandPoolCreateInfo pool_ci = {
            .flags = vk::CommandPoolCreateFlagBits::eTransient |
                     vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            .queueFamilyIndex = family,
        };
        pool = Vulkan::Check(device.createCommandPoolUnique(pool_ci));
        const vk::CommandBufferAllocateInfo alloc_info = {
            .commandPool = *pool,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = 1,
        };
        Vulkan::Check(device.allocateCommandBuffers(&alloc_info, &cmdbuf));
    }

    vk::Queue queue;
    /// The graphics queue is the scheduler's: its submits take the scheduler's submit lock.
    bool shared_queue;
    Vulkan::Semaphore timeline;
    vk::UniqueCommandPool pool;
    vk::CommandBuffer cmdbuf;
    Buffer staging;
    std::vector<vk::BufferCopy> copies;
    std::jthread thread;
};

ReadbackLanes::ReadbackLanes(const Vulkan::Instance& instance_, Vulkan::Scheduler& scheduler_,
                             BufferCache& buffer_cache_, u32 num_lanes, u64 window_size)
    : instance{instance_}, scheduler{scheduler_}, buffer_cache{buffer_cache_} {
    const auto copy_queues = instance.GetCopyQueues();
    const bool shared = copy_queues.empty();
    const u32 count =
        shared ? num_lanes : std::min(num_lanes, static_cast<u32>(copy_queues.size()));
    for (u32 i = 0; i < count; ++i) {
        const vk::Queue queue = shared ? instance.GetGraphicsQueue() : copy_queues[i];
        const u32 family =
            shared ? instance.GetGraphicsQueueFamilyIndex() : instance.GetCopyQueueFamilyIndex();
        auto& lane = lanes.emplace_back(
            std::make_unique<Lane>(instance, queue, family, shared, window_size));
        lane->thread =
            std::jthread([this, lane = lane.get()](std::stop_token stop) { Run(*lane, stop); });
    }
    LOG_INFO(Render_Vulkan, "Readback copy lanes: {} on {}", count,
             shared ? "the graphics queue" : "a transfer queue family");
}

ReadbackLanes::~ReadbackLanes() = default;

void ReadbackLanes::Copy(const Buffer* arena, VAddr window, u64 wait_tick, u64 copy_tick) {
    Job job{arena, window, wait_tick, copy_tick, false};
    std::unique_lock lk{mutex};
    pending.push_back(&job);
    job_cv.notify_one();
    done_cv.wait(lk, [&] { return job.done; });
}

void ReadbackLanes::Run(Lane& lane, std::stop_token stop) {
    Common::SetCurrentThreadName("shadPS4:Readback");
    while (true) {
        Job* job{};
        {
            std::unique_lock lk{mutex};
            Common::CondvarWait(job_cv, lk, stop, [this] { return !pending.empty(); });
            if (pending.empty()) {
                return;
            }
            job = pending.front();
            pending.erase(pending.begin());
        }
        Process(lane, *job);
        {
            std::scoped_lock lk{mutex};
            job->done = true;
        }
        done_cv.notify_all();
    }
}

void ReadbackLanes::Process(Lane& lane, const Job& job) {
    lane.copies.clear();
    const u64 total_size =
        buffer_cache.CollectRetiredWrites(job.arena, job.window, job.copy_tick, lane.copies);
    if (total_size == 0) {
        return;
    }
    const vk::CommandBufferBeginInfo begin_info = {
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    };
    Vulkan::Check(lane.cmdbuf.begin(begin_info));
    lane.cmdbuf.copyBuffer(job.arena->Handle(), lane.staging.Handle(), lane.copies);
    const vk::MemoryBarrier2 barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eHost,
        .dstAccessMask = vk::AccessFlagBits2::eHostRead,
    };
    lane.cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &barrier,
    });
    Vulkan::Check(lane.cmdbuf.end());

    // The GPU reached wait_tick before the owner handed the window over, so the wait is already
    // satisfied and legal on the graphics queue as well.
    const vk::Semaphore wait_semaphore = scheduler.GetWorkSemaphore()->Handle();
    const vk::Semaphore signal_semaphore = lane.timeline.Handle();
    const u64 signal_tick = lane.timeline.NextTick();
    const vk::PipelineStageFlags wait_stage = vk::PipelineStageFlagBits::eTransfer;
    const vk::TimelineSemaphoreSubmitInfo timeline_si = {
        .waitSemaphoreValueCount = 1,
        .pWaitSemaphoreValues = &job.wait_tick,
        .signalSemaphoreValueCount = 1,
        .pSignalSemaphoreValues = &signal_tick,
    };
    const vk::SubmitInfo submit_info = {
        .pNext = &timeline_si,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &wait_semaphore,
        .pWaitDstStageMask = &wait_stage,
        .commandBufferCount = 1,
        .pCommandBuffers = &lane.cmdbuf,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &signal_semaphore,
    };
    {
        std::unique_lock lk{Vulkan::Scheduler::submit_mutex, std::defer_lock};
        if (lane.shared_queue) {
            lk.lock();
        }
        Vulkan::Check(lane.queue.submit(submit_info));
    }
    lane.timeline.Wait(signal_tick);

    lane.staging.Invalidate(0, total_size);
    auto* memory = Core::Memory::Instance();
    for (const auto& copy : lane.copies) {
        memory->TryWriteBacking(std::bit_cast<u8*>(job.arena->CpuAddr() + copy.srcOffset),
                                lane.staging.mapped_data.data() + copy.dstOffset, copy.size);
    }
}

} // namespace VideoCore
