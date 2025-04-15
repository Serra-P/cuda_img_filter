//************************************************
//*                                              *
//*             (c) 2025 I. POTARD               *
//*                                              *
//*                                              *
//*                                              *
//************************************************

#include <cuda_runtime.h>
#include <fstream>
#include <iostream>

#include "image_io.hh"

namespace tifo
{

    __global__ void convert_rgb_bgr_kernel(uint8_t* pixels, int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size / 3)
        {
            int offset = idx * 3;
            uint8_t tmp = pixels[offset];
            pixels[offset] = pixels[offset + 2];
            pixels[offset + 2] = tmp;
        }
    }

    bool save_image_cuda(const rgb24_image& image, const char* filename)
    {
        tga_header header = { .idl_length = 0,
                              .color_map_type = 0,
                              .image_type = 2,
                              .cmap_start = 0,
                              .cmap_length = 0,
                              .cmap_depth = 0,
                              .x_offset = 0,
                              .y_offset = 0,
                              .width = static_cast<uint16_t>(image.data.sx),
                              .height = static_cast<uint16_t>(image.data.sy),
                              .pixel_depth = 24,
                              .image_descriptor = { .alpha_channel_bits = 0,
                                                    .image_origin = 0,
                                                    .unused = 0 } };

        uint8_t* host_buffer;
        cudaMallocHost(&host_buffer, image.data.length);

        cudaMemcpyAsync(host_buffer, image.data.d_pixels, image.data.length,
                        cudaMemcpyDeviceToHost);

        const int blockSize = 256;
        const int numBlocks =
            (image.data.length / 3 + blockSize - 1) / blockSize;
        convert_rgb_bgr_kernel<<<numBlocks, blockSize>>>(host_buffer,
                                                         image.data.length);

        cudaDeviceSynchronize();

        std::ofstream outfile(filename, std::ios::binary);
        if (!outfile.is_open())
        {
            std::cerr << "Error opening output file" << std::endl;
            cudaFreeHost(host_buffer);
            return false;
        }

        outfile.write(reinterpret_cast<char*>(&header), sizeof(tga_header));
        outfile.write(reinterpret_cast<char*>(host_buffer), image.data.length);
        outfile.close();

        cudaFreeHost(host_buffer);
        return true;
    }

    rgb24_image* load_image_cuda(const char* filename)
    {
        std::ifstream input(filename, std::ios::binary);
        if (!input.is_open())
        {
            std::cerr << "Error opening input file" << std::endl;
            return nullptr;
        }

        tga_header header;
        input.read(reinterpret_cast<char*>(&header), sizeof(tga_header));

        if (header.pixel_depth != 24)
        {
            std::cerr << "Only 24-bit TGA supported" << std::endl;
            return nullptr;
        }

        rgb24_image* image = new rgb24_image(header.width, header.height);

        input.read(reinterpret_cast<char*>(image->data.h_pixels.data()),
                   image->data.length);

        image->alloc_gpu();
        image->copy_to_gpu();

        const int blockSize = 256;
        const int numBlocks =
            (image->data.length / 3 + blockSize - 1) / blockSize;
        convert_rgb_bgr_kernel<<<numBlocks, blockSize>>>(image->data.d_pixels,
                                                         image->data.length);

        cudaDeviceSynchronize();
        return image;
    }

} // namespace tifo
