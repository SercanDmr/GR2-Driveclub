// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include <boost/container/small_vector.hpp>

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {
class Instance;
class Scheduler;
} // namespace Vulkan

namespace VideoCore {

class BufferCache;
struct Buffer;

/// Copy lanes serve the readback windows a direct read cannot: a lane thread copies the retired
/// GPU writes of a window from the arena into its staging buffer on its own queue and writes them
/// back to guest memory, so the faulting thread records no Vulkan commands and the GPU thread
/// never waits for the copy. The copies run on a transfer-only queue family when the device has
/// one, otherwise on the graphics queue under the scheduler's submit lock.
class ReadbackLanes {
public:
    explicit ReadbackLanes(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                           BufferCache& buffer_cache, u32 num_lanes, u64 window_size);
    ~ReadbackLanes();

    /// Copies the writes of the window stamped up to copy_tick into guest memory once the GPU has
    /// reached wait_tick, and returns when they landed. The caller owns the window.
    void Copy(const Buffer* arena, VAddr window, u64 wait_tick, u64 copy_tick);

private:
    struct Job {
        const Buffer* arena;
        VAddr window;
        u64 wait_tick;
        u64 copy_tick;
        bool done;
    };
    struct Lane;

    void Run(Lane& lane, std::stop_token stop);
    void Process(Lane& lane, const Job& job);

    const Vulkan::Instance& instance;
    Vulkan::Scheduler& scheduler;
    BufferCache& buffer_cache;
    std::mutex mutex;
    std::condition_variable_any job_cv;
    std::condition_variable done_cv;
    boost::container::small_vector<Job*, 8> pending;
    std::vector<std::unique_ptr<Lane>> lanes;
};

} // namespace VideoCore
