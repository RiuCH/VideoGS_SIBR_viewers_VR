// #pragma once

#include <cuda_runtime.h>
#include <stdint.h> // For uint8_t

// We must define the maximum possible attributes for pre-allocation.
// sh_degree=3 -> shs_dim=48.
// ply_dim = 14 + 48 = 62.
// num_att_index = ply_dim + 3 = 65.
#define MAX_ATTRIBUTES 65

// Define a reasonable max resolution for the attribute images.
// 2048*1024 = 2,097,152. Adjust if your images are larger.
#define MAX_IMAGE_PIXELS 2097152

// C-style wrapper to be called from GaussianView.cpp
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Launches the CUDA kernel to dequantize and reorganize all Gaussian data.
 *
 * @param blocks          Number of thread blocks to launch.
 * @param threads         Number of threads per block.
 * @param stream          CUDA stream for asynchronous execution.
 * @param m_count         The number of active Gaussians in this frame.
 * @param image_pixels    The number of pixels in each attribute image (rows * cols).
 * @param ply_dim         Total dimension of attributes (e.g., 17 for SH_degree=0).
 * @param shs_dim         Dimension of SHs (e.g., 3 for SH_degree=0).
 * @param d_raw_images    Device pointer to the buffer holding all raw uint8_t attribute images,
 * contiguously.
 * @param d_minmax_values Device pointer to the min/max table for each attribute.
 * @param d_pos_out       Device pointer to the output position buffer (float3).
 * @param d_rot_out       Device pointer to the output rotation buffer (float4).
 * @param d_scale_out     Device pointer to the output scale buffer (float3).
 * @param d_opacity_out   Device pointer to the output opacity buffer (float).
 * @param d_shs_out       Device pointer to the output SHs buffer (float*).
 * @param scale_factor    Scaling factor applied to position and scale attributes.
 */
void launch_dequantize_kernel(
    unsigned int blocks,
    unsigned int threads,
    cudaStream_t stream,
    int m_count,
    int image_pixels,
    int ply_dim,
    int shs_dim,
    int shs_dim_allocated,
    float scale_factor,
    const uint8_t* d_raw_images,
    const float* d_minmax_values,
    float* d_pos_out,
    float* d_rot_out,
    float* d_scale_out,
    float* d_opacity_out,
    float* d_shs_out
);

#ifdef __cplusplus
} // extern "C"
#endif