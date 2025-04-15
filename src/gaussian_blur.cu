#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>

#include "gaussian_blur.hh"
#include "utils.hh"

namespace tifo
{

    __device__ float gauss_weight(int distance, float sigma)
    {
        return expf(-0.5f * (distance / sigma) * (distance / sigma));
    }

    __device__ int clamp(int value, int min, int max)
    {
        return value < min ? min : (value > max ? max : value);
    }

    __global__ void gaussian_kernel(float* matrix, float sigma, int size,
                                    float* sum)
    {
        int center = size / 2;
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        int idy = threadIdx.y + blockIdx.y * blockDim.y;

        if (idx < size && idy < size)
        {
            int distance_x = idx - center;
            int distance_y = idy - center;
            float distance =
                sqrtf(distance_x * distance_x + distance_y * distance_y);
            float value = gauss_weight(distance, sigma);
            matrix[idy * size + idx] = value;
            atomicAdd(sum, value);
        }
    }

    std::vector<std::vector<float>> gaussian_matrix_cuda(float sigma)
    {
        int size = static_cast<int>(sigma * 6 + 1) | 1; // Matrice size
        std::vector<std::vector<float>> matrix(size, std::vector<float>(size));
        float h_sum = 0.0f;

        // Instantiate
        float* d_matrix;
        float* d_sum;
        CUDA_CHECK(cudaMalloc(&d_matrix, size * size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_sum, sizeof(float)));
        CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(float)));

        dim3 block(16, 16);
        dim3 grid((size + block.x - 1) / block.x,
                  (size + block.y - 1) / block.y);

        // GPU calculation of gaussian matrix
        gaussian_kernel<<<grid, block>>>(d_matrix, sigma, size, d_sum);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(
            cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));

        // Copy
        for (int i = 0; i < size; ++i)
        {
            CUDA_CHECK(cudaMemcpy(matrix[i].data(), d_matrix + i * size,
                                  size * sizeof(float),
                                  cudaMemcpyDeviceToHost));
        }

        // Normalize
        for (auto& row : matrix)
        {
            for (auto& val : row)
            {
                val /= h_sum;
            }
        }

        // Free
        CUDA_CHECK(cudaFree(d_matrix));
        CUDA_CHECK(cudaFree(d_sum));

        return matrix;
    }

    // Convolve with GPU
    __global__ void convolve_kernel(const uint8_t* image, uint8_t* result,
                                    int width, int height, const float* mask,
                                    int mask_size, int offset)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x < width && y < height)
        {
            float sum = 0.0f;
            for (int my = 0; my < mask_size; ++my)
            {
                for (int mx = 0; mx < mask_size; ++mx)
                {
                    int img_x = x + mx - offset;
                    int img_y = y + my - offset;
                    if (img_x >= 0 && img_x < width && img_y >= 0
                        && img_y < height)
                    {
                        sum += image[img_y * width + img_x]
                            * mask[my * mask_size + mx];
                    }
                }
            }
            result[y * width + x] =
                static_cast<uint8_t>(clamp(static_cast<int>(sum), 0, 255));
        }
    }

    // Initialize and call GPU calculation of convolve
    gray8_image convolve_gray_cuda(const gray8_image& image,
                                   const std::vector<std::vector<float>>& mask)
    {
        int mask_size = mask.size();
        int offset = mask_size / 2;
        gray8_image result(image.data.sx, image.data.sy);
        result.alloc_gpu();

        std::vector<float> flat_mask;
        flat_mask.reserve(mask_size * mask_size);
        for (const auto& row : mask)
        {
            flat_mask.insert(flat_mask.end(), row.begin(), row.end());
        }

        float* d_mask;
        CUDA_CHECK(cudaMalloc(&d_mask, mask_size * mask_size * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_mask, flat_mask.data(),
                              mask_size * mask_size * sizeof(float),
                              cudaMemcpyHostToDevice));

        dim3 block(16, 16);
        dim3 grid((image.data.sx + block.x - 1) / block.x,
                  (image.data.sy + block.y - 1) / block.y);

        convolve_kernel<<<grid, block>>>(image.gpu_ptr(), result.gpu_ptr(),
                                         image.data.sx, image.data.sy, d_mask,
                                         mask_size, offset);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(d_mask));

        result.copy_to_cpu();
        return result;
    }

    rgb24_image merge_canaux(const gray8_image& canal_r,
                             const gray8_image& canal_g,
                             const gray8_image& canal_b)
    {
        // Local non const copies
        gray8_image local_r = canal_r;
        gray8_image local_g = canal_g;
        gray8_image local_b = canal_b;

        rgb24_image result(local_r.data.sx, local_r.data.sy);
        result.alloc_gpu();

        // CPU data
        local_r.copy_to_cpu();
        local_g.copy_to_cpu();
        local_b.copy_to_cpu();

        // Merge
        for (size_t i = 0; i < local_r.data.length; ++i)
        {
            result.data.h_pixels[i * 3] = local_r.data.h_pixels[i];
            result.data.h_pixels[i * 3 + 1] = local_g.data.h_pixels[i];
            result.data.h_pixels[i * 3 + 2] = local_b.data.h_pixels[i];
        }

        // CPU data => GPU
        result.copy_to_gpu();
        return result;
    }

    rgb24_image gaussian_blur_cuda(const rgb24_image& image, float sigma)
    {
        // Local copy
        rgb24_image local_image = image;

        auto gaussian_mask = gaussian_matrix_cuda(sigma);

        gray8_image canal_r(local_image.data.sx, local_image.data.sy);
        gray8_image canal_g(local_image.data.sx, local_image.data.sy);
        gray8_image canal_b(local_image.data.sx, local_image.data.sy);

        // CPU data
        local_image.copy_to_cpu();

        // Separate RGB channels
        for (size_t i = 0; i < canal_r.data.length; ++i)
        {
            canal_r.data.h_pixels[i] = local_image.data.h_pixels[i * 3];
            canal_g.data.h_pixels[i] = local_image.data.h_pixels[i * 3 + 1];
            canal_b.data.h_pixels[i] = local_image.data.h_pixels[i * 3 + 2];
        }

        // CPU data => GPU
        canal_r.alloc_gpu();
        canal_g.alloc_gpu();
        canal_b.alloc_gpu();
        canal_r.copy_to_gpu();
        canal_g.copy_to_gpu();
        canal_b.copy_to_gpu();

        // Convolve on each
        auto result_r = convolve_gray_cuda(canal_r, gaussian_mask);
        auto result_g = convolve_gray_cuda(canal_g, gaussian_mask);
        auto result_b = convolve_gray_cuda(canal_b, gaussian_mask);

        // Merge
        return merge_canaux(result_r, result_g, result_b);
    }

} // namespace tifo
