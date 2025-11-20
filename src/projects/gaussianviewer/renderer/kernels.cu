#include "kernels.hpp"
#include <cuda_runtime.h>

/**
 * @brief Reads an 8-bit dequantized value.
 */
__device__ __forceinline__ float read_8bit_val(
    int idx,
    int attr_index,
    int image_pixels,
    const uint8_t* d_raw_images,
    const float* d_minmax_values)
{
    // pos (0-2) -> images 0-5
    // fdc (3-5) -> images 6-8 (skipped by original logic)
    // frest (6+) -> images 9+
    int image_index = attr_index + 3;

    // Read raw 8-bit value
    uint8_t raw_val = d_raw_images[image_index * image_pixels + idx];
    
    // Read min/max
    float min_val = d_minmax_values[attr_index * 2];
    float max_val = d_minmax_values[attr_index * 2 + 1];
    
    // Dequantize
    float val_f = static_cast<float>(raw_val) / 255.0f;
    return val_f * (max_val - min_val) + min_val;
}


/**
 * @brief Reads a 16-bit dequantized value (for position).
 */
__device__ __forceinline__ float read_16bit_val(
    int idx,
    int attr_index, // 0, 1, or 2
    int image_pixels,
    const uint8_t* d_raw_images,
    const float* d_minmax_values)
{
    // Position attributes use two 8-bit images each
    int image_index_even = attr_index * 2;
    int image_index_odd = attr_index * 2 + 1;

    // Read raw 8-bit values
    uint8_t even_val = d_raw_images[image_index_even * image_pixels + idx];
    uint8_t odd_val = d_raw_images[image_index_odd * image_pixels + idx];

    // Reconstruct 16-bit value
    uint16_t combined_val = (static_cast<uint16_t>(odd_val) << 8) | even_val;

    // Read min/max
    float min_val = d_minmax_values[attr_index * 2];
    float max_val = d_minmax_values[attr_index * 2 + 1];

    // Dequantize
    float val_f = static_cast<float>(combined_val) / 65535.0f;
    return val_f * (max_val - min_val) + min_val;
}


/**
 * @brief CUDA kernel to dequantize and reorganize all Gaussian data in parallel.
 */
__global__ void dequantize_and_reorganize_kernel(
    int m_count,
    int image_pixels,
    int ply_dim,
    int shs_dim,
    int shs_dim_allocated,
    float scale_factor,
    const uint8_t* d_raw_images,
    const float* d_minmax_values,
    float* d_pos_out,     // float3
    float* d_rot_out,     // float4
    float* d_scale_out,   // float3
    float* d_opacity_out, // float
    float* d_shs_out)     // float[shs_dim]
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m_count)
    {
        return;
    }

    // 1. Process Position (16-bit)
    // attr_index 0
    d_pos_out[idx * 3 + 0] = read_16bit_val(idx, 0, image_pixels, d_raw_images, d_minmax_values) * scale_factor; 
    // attr_index 1
    d_pos_out[idx * 3 + 1] = read_16bit_val(idx, 1, image_pixels, d_raw_images, d_minmax_values) * scale_factor; 
    // attr_index 2
    d_pos_out[idx * 3 + 2] = read_16bit_val(idx, 2, image_pixels, d_raw_images, d_minmax_values) * scale_factor; 
    // 2. Process SHs (8-bit)
    // Attributes start at 3 (fdc) and include all frest.
    int shs_base_attr = 6;
    
    // We calculate the base index using the *allocated* dimension (48)
    int shs_base_idx = idx * shs_dim_allocated; 

    // We loop only over the *active* dimensions (e.g., 3 for degree 0)
    for (int i = 0; i < shs_dim; ++i)
    {
        d_shs_out[shs_base_idx + i] = read_8bit_val(idx, shs_base_attr + i, image_pixels, d_raw_images, d_minmax_values);
    }

    // 3. Process Opacity (8-bit)
    //    Based on original C++: gaussian_data[ply_dim - 11] -> att_index = (ply_dim - 11) + 3 = ply_dim - 8
    int opacity_attr = ply_dim - 8;
    d_opacity_out[idx] = read_8bit_val(idx, opacity_attr, image_pixels, d_raw_images, d_minmax_values);

    // 4. Process Scale (8-bit)
    //    Based on original C++: gaussian_data[ply_dim - 10] -> att_index = (ply_dim - 10) + 3 = ply_dim - 7
    int scale_base_attr = ply_dim - 7;
    d_scale_out[idx * 3 + 0] = read_8bit_val(idx, scale_base_attr + 0, image_pixels, d_raw_images, d_minmax_values) * scale_factor;
    d_scale_out[idx * 3 + 1] = read_8bit_val(idx, scale_base_attr + 1, image_pixels, d_raw_images, d_minmax_values) * scale_factor;
    d_scale_out[idx * 3 + 2] = read_8bit_val(idx, scale_base_attr + 2, image_pixels, d_raw_images, d_minmax_values) * scale_factor;
    
    // 5. Process Rotation (8-bit)
    //    Based on original C++: gaussian_data[ply_dim - 7] -> att_index = (ply_dim - 7) + 3 = ply_dim - 4
    int rot_base_attr = ply_dim - 4;
    d_rot_out[idx * 4 + 0] = read_8bit_val(idx, rot_base_attr + 0, image_pixels, d_raw_images, d_minmax_values);
    d_rot_out[idx * 4 + 1] = read_8bit_val(idx, rot_base_attr + 1, image_pixels, d_raw_images, d_minmax_values);
    d_rot_out[idx * 4 + 2] = read_8bit_val(idx, rot_base_attr + 2, image_pixels, d_raw_images, d_minmax_values);
    d_rot_out[idx * 4 + 3] = read_8bit_val(idx, rot_base_attr + 3, image_pixels, d_raw_images, d_minmax_values);
}


// C-style launcher function
extern "C" void launch_dequantize_kernel(
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
    float* d_shs_out)
{
    dequantize_and_reorganize_kernel<<<blocks, threads, 0, stream>>>(
        m_count,
        image_pixels,
        ply_dim,
        shs_dim,
        shs_dim_allocated,
        scale_factor,
        d_raw_images,
        d_minmax_values,
        d_pos_out,
        d_rot_out,
        d_scale_out,
        d_opacity_out,
        d_shs_out
    );
}