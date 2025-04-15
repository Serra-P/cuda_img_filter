//************************************************
//*                                              *
//*             (c) 2025 I. POTARD               *
//*                                              *
//*                                              *
//*                                              *
//************************************************

#include <cuda_runtime.h>

#include "image.hh"

namespace tifo
{

    gray8_image::gray8_image(int sx, int sy)
    {
        data.sx = sx;
        data.sy = sy;
        data.length = sx * sy;
        data.h_pixels.resize(data.length);
        data.d_pixels = nullptr;
    }

    gray8_image::~gray8_image()
    {
        if (data.d_pixels)
        {
            cudaFree(data.d_pixels);
        }
    }

    void gray8_image::alloc_gpu()
    {
        if (!data.d_pixels)
        {
            cudaMalloc(&data.d_pixels, data.length * sizeof(uint8_t));
        }
    }

    void gray8_image::copy_to_gpu()
    {
        cudaMemcpy(data.d_pixels, data.h_pixels.data(),
                   data.length * sizeof(uint8_t), cudaMemcpyHostToDevice);
    }

    void gray8_image::copy_to_cpu()
    {
        cudaMemcpy(data.h_pixels.data(), data.d_pixels,
                   data.length * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    }

    void rgb24_image::copy_to_cpu()
    {
        cudaMemcpy2D(data.h_pixels.data(), data.sx * 3 * sizeof(uint8_t),
                     data.d_pixels, data.pitch, data.sx * 3 * sizeof(uint8_t),
                     data.sy, cudaMemcpyDeviceToHost);
    }

    rgb24_image::rgb24_image(int sx, int sy)
    {
        data.sx = sx;
        data.sy = sy;
        data.length = sx * sy * 3;
        data.h_pixels.resize(data.length);
        data.d_pixels = nullptr;
        data.pitch = 0;
    }

    rgb24_image::~rgb24_image()
    {
        if (data.d_pixels)
        {
            cudaFree(data.d_pixels);
        }
    }

    void rgb24_image::alloc_gpu()
    {
        if (!data.d_pixels)
        {
            cudaMallocPitch(&data.d_pixels, &data.pitch,
                            data.sx * 3 * sizeof(uint8_t), data.sy);
        }
    }

    void rgb24_image::copy_to_gpu()
    {
        cudaMemcpy2D(data.d_pixels, data.pitch, data.h_pixels.data(),
                     data.sx * 3 * sizeof(uint8_t),
                     data.sx * 3 * sizeof(uint8_t), data.sy,
                     cudaMemcpyHostToDevice);
    }

} // namespace tifo
