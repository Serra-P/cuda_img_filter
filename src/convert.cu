#include <cmath>
#include <cuda_runtime.h>

#include "convert.hh"
#include "utils.hh"

namespace tifo
{

    // RGB to Gray par le GPU pour paralleliser
    __global__ void rgb_to_gray_kernel(const uint8_t* __restrict__ rgb,
                                       uint8_t* __restrict__ gray, int width,
                                       int height)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x < width && y < height)
        {
            int rgb_idx = (y * width + x) * 3;
            int gray_idx = y * width + x;

            gray[gray_idx] = 0.299f * rgb[rgb_idx] + 0.587f * rgb[rgb_idx + 1]
                + 0.114f * rgb[rgb_idx + 2];
        }
    }

    // Appel paralleliser de la fonction rgb to gray du gpu
    gray8_image to_gray_cuda(const rgb24_image& img)
    {
        gray8_image result(img.data.sx, img.data.sy);
        result.alloc_gpu();

        dim3 blockSize(16, 16);
        dim3 gridSize((img.data.sx + blockSize.x - 1) / blockSize.x,
                      (img.data.sy + blockSize.y - 1) / blockSize.y);

        rgb_to_gray_kernel<<<gridSize, blockSize>>>(
            img.data.d_pixels, result.data.d_pixels, img.data.sx, img.data.sy);

        result.copy_to_cpu();
        return result;
    }

    // Utilisation du GPU pour transformer le gray en rgb
    __global__ void gray_to_rgb_kernel(const uint8_t* __restrict__ gray,
                                       uint8_t* __restrict__ rgb, int width,
                                       int height)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x < width && y < height)
        {
            int gray_idx = y * width + x;
            int rgb_idx = gray_idx * 3;
            uint8_t val = gray[gray_idx];

            rgb[rgb_idx] = val;
            rgb[rgb_idx + 1] = val;
            rgb[rgb_idx + 2] = val;
        }
    }

    // Appel paralleliser de la fonction gray to rgb du gpu
    rgb24_image to_rgb_cuda(const gray8_image& img)
    {
        rgb24_image result(img.data.sx, img.data.sy);
        result.alloc_gpu();

        dim3 blockSize(16, 16);
        dim3 gridSize((img.data.sx + blockSize.x - 1) / blockSize.x,
                      (img.data.sy + blockSize.y - 1) / blockSize.y);

        gray_to_rgb_kernel<<<gridSize, blockSize>>>(
            img.data.d_pixels, result.data.d_pixels, img.data.sx, img.data.sy);

        result.copy_to_cpu();
        return result;
    }

    // Sauvegarde ce tableau en constante pour de l optimisation
    __constant__ float hsv_constants[6][3] = { { 1, 0, 0 }, { 0, 1, 0 },
                                               { 0, 0, 1 }, { 1, 1, 0 },
                                               { 0, 1, 1 }, { 1, 0, 1 } };

    // GPU pour conversion RGB vers HSV
    __global__ void rgb_to_hsv_kernel(const uint8_t* rgb, HSV* hsv, int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size)
        {
            int rgb_idx = idx * 3;
            float r = rgb[rgb_idx] / 255.0f;
            float g = rgb[rgb_idx + 1] / 255.0f;
            float b = rgb[rgb_idx + 2] / 255.0f;

            float cmax = fmaxf(r, fmaxf(g, b));
            float cmin = fminf(r, fminf(g, b));
            float delta = cmax - cmin;

            hsv[idx].v = cmax;

            if (delta < 1e-5f)
            {
                hsv[idx].h = 0;
                hsv[idx].s = 0;
            }
            else
            {
                hsv[idx].s = delta / cmax;

                if (cmax == r)
                {
                    hsv[idx].h = 60.0f * fmodf((g - b) / delta, 6.0f);
                }
                else if (cmax == g)
                {
                    hsv[idx].h = 60.0f * (((b - r) / delta) + 2.0f);
                }
                else
                {
                    hsv[idx].h = 60.0f * (((r - g) / delta) + 4.0f);
                }

                if (hsv[idx].h < 0)
                    hsv[idx].h += 360.0f;
            }
        }
    }

    // GPU pour fusion des canaux
    __global__ void merge_channels_kernel(const uint8_t* r, const uint8_t* g,
                                          const uint8_t* b, uint8_t* rgb,
                                          int width, int height)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x < width && y < height)
        {
            int idx = y * width + x;
            int rgb_idx = idx * 3;

            rgb[rgb_idx] = r[idx];
            rgb[rgb_idx + 1] = g[idx];
            rgb[rgb_idx + 2] = b[idx];
        }
    }

    // Appel paralleliser de la fonction merge
    rgb24_image merge_canaux_cuda(const gray8_image& canal_r,
                                  const gray8_image& canal_g,
                                  const gray8_image& canal_b)
    {
        rgb24_image result(canal_r.data.sx, canal_r.data.sy);
        result.alloc_gpu();

        dim3 blockSize(16, 16);
        dim3 gridSize((canal_r.data.sx + blockSize.x - 1) / blockSize.x,
                      (canal_r.data.sy + blockSize.y - 1) / blockSize.y);

        merge_channels_kernel<<<gridSize, blockSize>>>(
            canal_r.data.d_pixels, canal_g.data.d_pixels, canal_b.data.d_pixels,
            result.data.d_pixels, canal_r.data.sx, canal_r.data.sy);

        result.copy_to_cpu();
        return result;
    }

} // namespace tifo
