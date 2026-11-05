// Aligned allocation for tensor buffers, with usage counters and a cache of
// freed blocks.
//
// Asking the OS for a large block is the slowest thing a tensor operation can
// do: every page faults on first use, which can take longer than the arithmetic.
// Code like `x = a + b` in a loop would hit this each time around, because the
// allocator hands large blocks back to the OS as soon as they're freed. So large
// blocks are kept here when freed, and the next tensor of a similar size reuses
// one without any page faults.
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>

#if defined(__linux__)
#include <sys/mman.h>
#endif

#include "minitensor/runtime.hpp"

namespace minitensor {
namespace {

std::atomic<std::size_t> g_live_bytes{0};
std::atomic<std::size_t> g_peak_bytes{0};
std::atomic<std::size_t> g_allocations{0};

// Creating the thread team takes milliseconds. Doing it for the first tensor of
// this size means it normally happens before the work that a program times.
constexpr std::size_t kStartThreadsBytes = std::size_t{1} << 18;
// Small blocks are left to malloc, which is already fast for them.
constexpr std::size_t kMinCachedBytes = std::size_t{1} << 16;
// The cache keeps at most this much memory from being returned to the OS.
constexpr std::size_t kMaxCachedBytes = std::size_t{1} << 30;
constexpr std::size_t kBlocksPerClass = 8;
constexpr unsigned kSizeClasses = 48;

// Stored just before each aligned block.
struct BlockHeader {
    void* raw;            // What malloc returned.
    std::size_t bytes;    // What the caller asked for.
    unsigned size_class;  // kSizeClasses if the block isn't cacheable.
};

constexpr std::size_t kOverhead = kAlignment + sizeof(BlockHeader);

BlockHeader* header_of(void* block) noexcept {
    return reinterpret_cast<BlockHeader*>(static_cast<unsigned char*>(block) - sizeof(BlockHeader));
}

// Blocks are rounded up to a power of two so that close sizes can share them.
// The unused end of a block is never touched, so it uses no physical memory.
unsigned size_class_of(std::size_t bytes) noexcept {
    if (bytes < kMinCachedBytes) return kSizeClasses;
    const unsigned size_class = 64u - static_cast<unsigned>(__builtin_clzll(bytes - 1));
    return size_class < kSizeClasses ? size_class : kSizeClasses;
}

std::size_t capacity_of(unsigned size_class, std::size_t bytes) noexcept {
    return size_class < kSizeClasses ? std::size_t{1} << size_class : bytes;
}

class BlockCache {
public:
    void* take(unsigned size_class) noexcept {
        if (size_class == kSizeClasses) return nullptr;
        const std::lock_guard<std::mutex> lock(mutex_);
        Shelf& shelf = shelves_[size_class];
        if (shelf.count == 0) return nullptr;
        total_bytes_ -= std::size_t{1} << size_class;
        return shelf.blocks[--shelf.count];
    }

    bool put(void* block, unsigned size_class) noexcept {
        if (size_class == kSizeClasses) return false;
        const std::size_t capacity = std::size_t{1} << size_class;
        const std::lock_guard<std::mutex> lock(mutex_);
        Shelf& shelf = shelves_[size_class];
        if (shelf.count == kBlocksPerClass || total_bytes_ + capacity > kMaxCachedBytes)
            return false;
        shelf.blocks[shelf.count++] = block;
        total_bytes_ += capacity;
        return true;
    }

    void clear() noexcept {
        const std::lock_guard<std::mutex> lock(mutex_);
        for (Shelf& shelf : shelves_) {
            while (shelf.count > 0) std::free(header_of(shelf.blocks[--shelf.count])->raw);
        }
        total_bytes_ = 0;
    }

private:
    struct Shelf {
        std::array<void*, kBlocksPerClass> blocks{};
        std::size_t count = 0;
    };

    std::mutex mutex_;
    std::array<Shelf, kSizeClasses> shelves_{};
    std::size_t total_bytes_ = 0;
};

// Never destroyed, because a tensor with static storage may be freed at any
// point during program shutdown.
BlockCache& block_cache() noexcept {
    static BlockCache* cache = new BlockCache;
    return *cache;
}

void track_allocation(std::size_t bytes) noexcept {
    const std::size_t live = g_live_bytes.fetch_add(bytes, std::memory_order_relaxed) + bytes;
    std::size_t peak = g_peak_bytes.load(std::memory_order_relaxed);
    while (live > peak &&
           !g_peak_bytes.compare_exchange_weak(peak, live, std::memory_order_relaxed)) {
    }
    g_allocations.fetch_add(1, std::memory_order_relaxed);
}

// With 4 KiB pages, walking a large tensor keeps missing the CPU's table of
// recent page translations. Asking for 2 MiB pages makes those misses rare. The
// OS may ignore this, which is harmless.
void advise_huge_pages(void* raw, std::size_t bytes) noexcept {
#if defined(__linux__) && defined(MADV_HUGEPAGE)
    constexpr std::size_t kHugePage = std::size_t{1} << 21;
    if (bytes < kHugePage) return;
    constexpr std::uintptr_t kPageMask = 4095;
    const auto start = (reinterpret_cast<std::uintptr_t>(raw) + kPageMask) & ~kPageMask;
    const auto end = (reinterpret_cast<std::uintptr_t>(raw) + bytes) & ~kPageMask;
    if (end > start) madvise(reinterpret_cast<void*>(start), end - start, MADV_HUGEPAGE);
#else
    (void)raw;
    (void)bytes;
#endif
}

void* align_block(void* raw, std::size_t bytes, unsigned size_class) {
    if (raw == nullptr) throw std::bad_alloc();
    advise_huge_pages(raw, bytes);
    const auto first_free = reinterpret_cast<std::uintptr_t>(raw) + sizeof(BlockHeader);
    const std::uintptr_t mask = kAlignment - 1;
    auto* aligned = reinterpret_cast<unsigned char*>((first_free + mask) & ~mask);
    new (aligned - sizeof(BlockHeader)) BlockHeader{raw, bytes, size_class};
    track_allocation(bytes);
    return aligned;
}

}  // namespace

MemoryStats memory_stats() noexcept {
    return {g_live_bytes.load(std::memory_order_relaxed),
            g_peak_bytes.load(std::memory_order_relaxed),
            g_allocations.load(std::memory_order_relaxed)};
}

void release_cached_memory() noexcept {
    block_cache().clear();
}

namespace detail {

void* aligned_malloc(std::size_t bytes) {
    if (bytes == 0) return nullptr;
    if (bytes > SIZE_MAX - kOverhead) throw std::bad_alloc();
    if (bytes >= kStartThreadsBytes) start_threads_once();

    const unsigned size_class = size_class_of(bytes);
    if (void* block = block_cache().take(size_class)) {
        header_of(block)->bytes = bytes;
        track_allocation(bytes);
        return block;
    }
    return align_block(std::malloc(capacity_of(size_class, bytes) + kOverhead), bytes, size_class);
}

// A new large block from calloc is free until its pages are used, because the OS
// supplies them already zeroed. That makes a big tensor of zeros cheap to create,
// and is why this doesn't look in the cache.
void* aligned_calloc(std::size_t bytes) {
    if (bytes == 0) return nullptr;
    if (bytes > SIZE_MAX - kOverhead) throw std::bad_alloc();
    if (bytes >= kStartThreadsBytes) start_threads_once();

    const unsigned size_class = size_class_of(bytes);
    return align_block(std::calloc(1, capacity_of(size_class, bytes) + kOverhead), bytes,
                       size_class);
}

void aligned_free(void* block) noexcept {
    if (block == nullptr) return;
    const BlockHeader* header = header_of(block);
    g_live_bytes.fetch_sub(header->bytes, std::memory_order_relaxed);
    if (!block_cache().put(block, header->size_class)) std::free(header->raw);
}

}  // namespace detail
}  // namespace minitensor
