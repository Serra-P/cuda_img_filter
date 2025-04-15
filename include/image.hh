#ifndef IMAGE_HH
#define IMAGE_HH

#include <cstdint>
#include <vector>

#define TL_IMAGE_ALIGNMENT 64

namespace tifo
{

    struct gray8_data
    {
        uint8_t* d_pixels; // Données GPU
        std::vector<uint8_t> h_pixels; // Données CPU
        int sx, sy;
        size_t length;
    };

    struct rgb24_data
    {
        uint8_t* d_pixels; // Données GPU
        std::vector<uint8_t> h_pixels; // Données CPU
        int sx, sy;
        size_t length;
        size_t pitch;
    };

    class gray8_image
    {
    public:
        gray8_data data;

        __host__ gray8_image(int sx, int sy);
        __host__ ~gray8_image();

        __host__ void alloc_gpu();
        __host__ void free_gpu();
        __host__ void copy_to_gpu();
        __host__ void copy_to_cpu();

        __host__ __device__ uint8_t* gpu_ptr()
        {
            return data.d_pixels;
        }
        __host__ __device__ const uint8_t* gpu_ptr() const
        {
            return data.d_pixels;
        }
    };

    class rgb24_image
    {
    public:
        rgb24_data data;

        __host__ rgb24_image(int sx, int sy);
        __host__ ~rgb24_image();

        __host__ void alloc_gpu();
        __host__ void free_gpu();
        __host__ void copy_to_gpu();
        __host__ void copy_to_cpu();

        __host__ __device__ uint8_t* gpu_ptr()
        {
            return data.d_pixels;
        }
        __host__ __device__ const uint8_t* gpu_ptr() const
        {
            return data.d_pixels;
        }
    };

} // namespace tifo
#endif
