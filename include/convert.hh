#ifndef CONVERT_HH
#define CONVERT_HH

#include <cuda_runtime.h>

#include "image.hh"

namespace tifo
{

    // Structure de couleurs HSV
    struct HSV
    {
        float h; // Teinte (0-360)
        float s; // Saturation (0-1)
        float v; // Valeur (0-1)
    };

    /**
     * Conversion RGB vers niveaux de gris (GPU)
     * @param img Image RGB d'entrée
     * @return Image en niveaux de gris
     */
    gray8_image to_gray_cuda(const rgb24_image& img);

    /**
     * Conversion niveaux de gris vers RGB (GPU)
     * @param img Image en niveaux de gris
     * @return Image RGB
     */
    rgb24_image to_rgb_cuda(const gray8_image& img);

    /**
     * Fusion de 3 canaux en une image RGB (GPU)
     * @param canal_r Canal rouge
     * @param canal_g Canal vert
     * @param canal_b Canal bleu
     * @return Image RGB combinée
     */
    rgb24_image merge_canaux_cuda(const gray8_image& canal_r,
                                  const gray8_image& canal_g,
                                  const gray8_image& canal_b);

    // Declaration externe
    __global__ void rgb_to_gray_kernel(const uint8_t* __restrict__ rgb,
                                       uint8_t* __restrict__ gray, int width,
                                       int height);

    // Declaration externe
    __global__ void gray_to_rgb_kernel(const uint8_t* __restrict__ gray,
                                       uint8_t* __restrict__ rgb, int width,
                                       int height);

    // Declaration externe
    __global__ void rgb_to_hsv_kernel(const uint8_t* rgb, HSV* hsv, int size);

    // Declaration externe
    __global__ void merge_channels_kernel(const uint8_t* r, const uint8_t* g,
                                          const uint8_t* b, uint8_t* rgb,
                                          int width, int height);
    gray8_image adjust_contrast_cuda(const gray8_image& img);
} // namespace tifo

#endif // CONVERTION_HH
