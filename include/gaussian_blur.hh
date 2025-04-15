#ifndef GAUSSIAN_BLUR_HH
#define GAUSSIAN_BLUR_HH

#include <cuda_runtime.h>
#include <vector>

#include "image_io.hh"

namespace tifo
{
    // Return gaussian matrix with a given sigma
    std::vector<std::vector<float>> gaussian_matrix(float sigma);

    // Return image convolve with a gaussian matrix generate with given sigma
    rgb24_image gaussian_blur_cuda(const rgb24_image& image, float sigma);

} // namespace tifo

#endif // GAUSSIAN_BLUR_HH
