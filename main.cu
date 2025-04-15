#include <cuda_runtime.h>
#include <iostream>

#include "convert.hh"
#include "gaussian_blur.hh"
#include "image.hh"
#include "image_io.hh"
#include "utils.hh"

using namespace tifo;
int main(int argc, char** argv)
{
    if (argc != 3)
        return !printf("Usage: %s input.tga output.tga\n", argv[0]);

    tifo::rgb24_image* img = tifo::load_image_cuda(argv[1]);
    if (!img)
        return EXIT_FAILURE;

    tifo::rgb24_image rgb_img = tifo::gaussian_blur_cuda(*img, 3.0);
    tifo::save_image_cuda(rgb_img, argv[2]);

    delete img;
    cudaDeviceReset();

    printf("Traitement terminé avec succès !\n");
    printf("Image d'entrée : %s\n", argv[1]);
    printf("Résultat sauvegardé dans : %s\n", argv[2]);
    return EXIT_SUCCESS;
}
