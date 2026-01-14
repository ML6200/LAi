#ifndef LAI_CORE_ALLOCATOR_H
#define LAI_CORE_ALLOCATOR_H

#include "types.h"
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cassert>

namespace lai {

// Simple arena allocator for zero-allocation inference
class Arena {
public:
    explicit Arena(size_t size = 64 * 1024 * 1024)  // 64MB default
        : buffer_(nullptr), size_(size), offset_(0) {
        buffer_ = static_cast<u8*>(std::aligned_alloc(64, size));
        if (!buffer_) {
            throw std::bad_alloc();
        }
    }

    ~Arena() {
        if (buffer_) {
            std::free(buffer_);
        }
    }

    // Non-copyable
    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;

    // Movable
    Arena(Arena&& other) noexcept
        : buffer_(other.buffer_), size_(other.size_), offset_(other.offset_) {
        other.buffer_ = nullptr;
        other.size_ = 0;
        other.offset_ = 0;
    }

    Arena& operator=(Arena&& other) noexcept {
        if (this != &other) {
            if (buffer_) std::free(buffer_);
            buffer_ = other.buffer_;
            size_ = other.size_;
            offset_ = other.offset_;
            other.buffer_ = nullptr;
            other.size_ = 0;
            other.offset_ = 0;
        }
        return *this;
    }

    // Allocate aligned memory from arena
    void* alloc(size_t bytes, size_t alignment = 64) {
        // Align offset
        size_t aligned_offset = (offset_ + alignment - 1) & ~(alignment - 1);

        if (aligned_offset + bytes > size_) {
            return nullptr;  // Out of memory
        }

        void* ptr = buffer_ + aligned_offset;
        offset_ = aligned_offset + bytes;
        return ptr;
    }

    // Allocate and zero-initialize
    void* calloc(size_t bytes, size_t alignment = 64) {
        void* ptr = alloc(bytes, alignment);
        if (ptr) {
            std::memset(ptr, 0, bytes);
        }
        return ptr;
    }

    // Typed allocation
    template<typename T>
    T* alloc(size_t count = 1) {
        return static_cast<T*>(alloc(count * sizeof(T), alignof(T)));
    }

    // Reset arena (doesn't free memory, just resets offset)
    void reset() {
        offset_ = 0;
    }

    // Get current usage
    size_t used() const { return offset_; }
    size_t capacity() const { return size_; }
    size_t available() const { return size_ - offset_; }

    // Save/restore position for temporary allocations
    size_t save() const { return offset_; }
    void restore(size_t pos) { offset_ = pos; }

private:
    u8* buffer_;
    size_t size_;
    size_t offset_;
};

// RAII scope guard for arena
class ArenaScope {
public:
    explicit ArenaScope(Arena& arena) : arena_(arena), saved_(arena.save()) {}
    ~ArenaScope() { arena_.restore(saved_); }

    ArenaScope(const ArenaScope&) = delete;
    ArenaScope& operator=(const ArenaScope&) = delete;

private:
    Arena& arena_;
    size_t saved_;
};

// Pool allocator for fixed-size objects
template<typename T, size_t BlockSize = 4096>
class Pool {
public:
    Pool() : free_list_(nullptr) {}

    ~Pool() {
        for (auto* block : blocks_) {
            std::free(block);
        }
    }

    Pool(const Pool&) = delete;
    Pool& operator=(const Pool&) = delete;

    T* alloc() {
        if (!free_list_) {
            allocate_block();
        }
        Node* node = free_list_;
        free_list_ = node->next;
        return reinterpret_cast<T*>(node);
    }

    void free(T* ptr) {
        Node* node = reinterpret_cast<Node*>(ptr);
        node->next = free_list_;
        free_list_ = node;
    }

private:
    struct Node {
        Node* next;
    };

    void allocate_block() {
        static_assert(sizeof(T) >= sizeof(Node), "T must be at least pointer-sized");

        constexpr size_t obj_size = sizeof(T) > sizeof(Node) ? sizeof(T) : sizeof(Node);
        constexpr size_t count = BlockSize / obj_size;

        u8* block = static_cast<u8*>(std::aligned_alloc(alignof(T), count * obj_size));
        if (!block) throw std::bad_alloc();

        blocks_.push_back(block);

        // Link all objects in block to free list
        for (size_t i = 0; i < count; ++i) {
            Node* node = reinterpret_cast<Node*>(block + i * obj_size);
            node->next = free_list_;
            free_list_ = node;
        }
    }

    Node* free_list_;
    std::vector<u8*> blocks_;
};

// Global scratch arena for temporary allocations
inline Arena& scratch_arena() {
    static Arena arena(32 * 1024 * 1024);  // 32MB scratch
    return arena;
}

} // namespace lai

#endif // LAI_CORE_ALLOCATOR_H
