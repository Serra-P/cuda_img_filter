#ifndef IMAGE_IO_HH
#define IMAGE_IO_HH

#include <cstdio>

#include "image.hh"

namespace tifo
{

#pragma pack(push, 1)
    struct tga_header
    {
        uint8_t idl_length;
        uint8_t color_map_type;
        uint8_t image_type;
        uint16_t cmap_start;
        uint16_t cmap_length;
        uint8_t cmap_depth;
        uint16_t x_offset;
        uint16_t y_offset;
        uint16_t width;
        uint16_t height;
        uint8_t pixel_depth;
        struct
        {
            uint8_t alpha_channel_bits : 4;
            uint8_t image_origin : 2;
            uint8_t unused : 2;
        } image_descriptor;
    };
#pragma pack(pop)

    // CPU
    __host__ bool save_image(const rgb24_image& image, const char* filename);
    // CPU
    __host__ rgb24_image* load_image(const char* filename);
    // CPU
    __host__ bool save_image_cuda(const rgb24_image& image,
                                  const char* filename);
    // CPU
    __host__ rgb24_image* load_image_cuda(const char* filename);

    // GPU
    __global__ void convert_rgb_bgr_kernel(uint8_t* pixels, int size);

} // namespace tifo

#endif // IMAGE_IO_HH
