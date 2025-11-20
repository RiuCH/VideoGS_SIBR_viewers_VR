/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr
 */

#include <projects/gaussianviewer/renderer/GaussianView.hpp>
#include <core/graphics/GUI.hpp>
#include <thread>
#include <boost/asio.hpp>
#include <rasterizer.h>
#include <imgui_internal.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <picojson/picojson.hpp>
#include <chrono>
#include <opencv2/core/utils/logger.hpp>
#include <imgui/imgui.h>
#include <projects/gaussianviewer/renderer/JsonUtils.hpp>
#include <projects/gaussianviewer/renderer/OpenCVVideoDecoder.hpp>
#include <projects/gaussianviewer/renderer/GSVideoDecoder.hpp>
#include <future>
#include <execution>
#include <bitset>
#include <fstream>
#include <string>
#include <sstream>
#include <npp.h>
#include <nppi.h>

typedef sibr::Vector3f Pos;
template<int D>
struct SHs
{
	float shs[(D+1)*(D+1)*3];
};
struct Scale
{
	float scale[3];
};
struct Rot
{
	float rot[4];
};
template<int D>
struct RichPoint
{
	Pos pos;
	float n[3]; // normals, unused in splatting but often in PLY
	SHs<D> shs;
	float opacity;
	Scale scale;
	Rot rot;
};

// Define a structure for the PLY data we expect for background
// This matches the ordering in many 3DGS PLY files.
// SH degree is assumed to be 3.
struct PlyGaussian
{
    float x, y, z;
    float nx, ny, nz;
    float f_dc_0, f_dc_1, f_dc_2;
    float f_rest[45]; // SHs rest
    float opacity;
    float scale_0, scale_1, scale_2;
    float rot_0, rot_1, rot_2, rot_3;
};

float sigmoid(const float m1)
{
	return 1.0f / (1.0f + exp(-m1));
}

# define CUDA_SAFE_CALL_ALWAYS(A) \
A; \
cudaDeviceSynchronize(); \
if (cudaPeekAtLastError() != cudaSuccess) \
SIBR_ERR << cudaGetErrorString(cudaGetLastError());

#if DEBUG || _DEBUG
# define CUDA_SAFE_CALL(A) CUDA_SAFE_CALL_ALWAYS(A)
#else
# define CUDA_SAFE_CALL(A) A
#endif

bool image_dequan(const cv::Mat& m_att_img, std::vector<float>& gaussian_vector, float max, float min) {
    cv::Mat att_img;
    m_att_img.convertTo(att_img, CV_32F, 1.0 / (std::pow(2.0, 8) - 1.0));
	// perform dequantization
	// att_img.convertTo(att_img, CV_32F, 1.0 / (std::pow(2.0, 8) - 1.0)); // Moved up
	cv::Mat dequantized_img = att_img * (max - min) + min;
	// convert to 1D vector
	std::vector<float> deimg_vector(dequantized_img.rows * dequantized_img.cols);
	if (dequantized_img.isContinuous()) {
		deimg_vector.assign((float*)dequantized_img.datastart, (float*)dequantized_img.dataend);
	} else {
		return false;
	}
	gaussian_vector = deimg_vector;
	return true;
}

// Load the Gaussians from the given png.
void sibr::GaussianView::loadVideo_func(int frame_index)
{
	int json_index = frame_index;
	int shs_dim = 3 * (_sh_degree + 1) * (_sh_degree + 1);
	int ply_dim = (14 + shs_dim);

	const int shs_dim_allocated = 3 * (3 + 1) * (3 + 1);

	// get info from json
	picojson::object frameobj = minmax_obj[std::to_string(json_index)].get<picojson::object>();
	int m_count = static_cast<int>(frameobj["num"].get<double>());
	picojson::array arr = frameobj["info"].get<picojson::array>();
	std::vector<float> minmax_values; // This is the only CPU data we process
	for (picojson::value& val : arr) {
		float value = static_cast<float>(val.get<double>());
		minmax_values.push_back(value);
	}
	if (minmax_values.size() != 2 * ply_dim) {
		SIBR_ERR << "Error: " << "vector size not match" << std::endl;
        return;
	}

    // Check for buffer overflow
    if (m_count > MAX_GAUSSIANS_PER_FRAME) {
        SIBR_ERR << "ERROR: Frame " << frame_index << " has " << m_count << " Gaussians, "
                 << "which is more than the pre-allocated buffer size of " 
                 << MAX_GAUSSIANS_PER_FRAME << "!" << std::endl;
        return;
    }

    // All attribute images *must* have the same size.
    int image_pixels = global_png_vector[0][frame_index].total(); // rows * cols
    if (image_pixels > MAX_IMAGE_PIXELS) {
        SIBR_ERR << "ERROR: Frame " << frame_index << " has images of size " << image_pixels << " pixels, "
                 << "which is more than the pre-allocated buffer size of "
                 << MAX_IMAGE_PIXELS << "!" << std::endl;
        return;
    }
    int num_att_to_copy = ply_dim + 3; // This is num_att_index
    if (num_att_to_copy > MAX_ATTRIBUTES) {
         SIBR_ERR << "ERROR: Frame " << frame_index << " requires " << num_att_to_copy << " attributes, "
                 << "which is more than the pre-allocated buffer size of "
                 << MAX_ATTRIBUTES << "!" << std::endl;
        return;
    }

	int slot = frame_index % GPU_RING_BUFFER_SLOTS;
    cudaStream_t stream = data_streams[slot];
    GpuFrameSlot& current_slot = gpu_ring_buffer[slot];

    auto start = std::chrono::high_resolution_clock::now();
    
    // Copy min/max table
    CUDA_SAFE_CALL(cudaMemcpyAsync(current_slot.minmax_values_cuda, minmax_values.data(), 
                                   minmax_values.size() * sizeof(float), cudaMemcpyHostToDevice, stream));

    // Copy all raw attribute images contiguously
    uint8_t* d_raw_image_ptr = current_slot.raw_attributes_cuda;
    for (int i = 0; i < num_att_to_copy; ++i)
    {
        cv::Mat& img = global_png_vector[i][frame_index];
        if (img.total() != image_pixels) {
            SIBR_ERR << "ERROR: Attribute image " << i << " has mismatched size!";
            // Handle error...
        }
        CUDA_SAFE_CALL(cudaMemcpyAsync(d_raw_image_ptr, img.data, 
                                       image_pixels * sizeof(uint8_t), cudaMemcpyHostToDevice, stream));
        d_raw_image_ptr += image_pixels * sizeof(uint8_t);
    }

    unsigned int threads = 256;
    unsigned int blocks = (m_count + threads - 1) / threads;

    launch_dequantize_kernel(
        blocks,
        threads,
        stream,
        m_count,
        image_pixels,
        ply_dim,
        shs_dim,
		shs_dim_allocated,
        current_slot.raw_attributes_cuda,
        current_slot.minmax_values_cuda,
        current_slot.pos_cuda,
        current_slot.rot_cuda,
        current_slot.scale_cuda,
        current_slot.opacity_cuda,
        current_slot.shs_cuda
    );
    
    // Memset the rect buffer (still needed for culling)
	CUDA_SAFE_CALL(cudaMemsetAsync(current_slot.rect_cuda, 0, 2 * m_count * sizeof(int), stream));
	
    P_array[frame_index] = m_count;
    
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	// std::cout << frame_index << "CUDA Kernel Launch Elapsed time: " << elapsed.count() << " ms" << std::endl;

    // Record an event in the stream. The render thread will wait for this.
    CUDA_SAFE_CALL(cudaEventRecord(data_events[slot], stream));

	ready_array[frame_index] = 1;
	ready_frames = frame_index;
	
}

// ready gaussian
void sibr::GaussianView::readyVideo_func() {
	int i;
	while (frame_changed == false) {
		int i = -1;
		{
			std::unique_lock<std::mutex> lock(mtx_ready);
			cv_ready.wait(lock, [this] { return !need_ready_q.empty() || frame_changed; });
			if (frame_changed) {
				std::cout << "[readyVideo_func] frame_changed signal received. Exiting." << std::endl;
				break;
			}

			i = need_ready_q.front();
			need_ready_q.pop();
		} 

		// Now, perform the check and the loading without holding the lock
		int current_frame_id = 0;
		{
			std::lock_guard<std::mutex> frame_lock(mtx_frame_id);
			current_frame_id = frame_id;
		}

		if ( i >= current_frame_id + ready_cache_size ) {
			std::cout << "[readyVideo_func] Frame " << i << " is out of window size, skipping." << std::endl;
			continue;
		}
		if (ready_array[i] == 1) {
			std::cout << "[readyVideo_func] Frame " << i << " is already ready, skipping." << std::endl;
			continue;
		}

		if (downloaded_array[i] == 0) {
			std::cout << "[readyVideo_func] Frame " << i << " not downloaded yet. Re-queueing and sleeping." << std::endl;
			// Put the frame back in the queue to be processed later
			{
				std::lock_guard<std::mutex> lock(mtx_ready);
				need_ready_q.push(i);
			}
			// Sleep to prevent this thread from spin-locking on this frame
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
			continue;
		}
		
		std::cout << "[readyVideo_func] Preparing to load frame " << i << std::endl;
		loadVideo_func(i);
		std::cout << "[readyVideo_func] Frame " << i << " is now ready." << std::endl;
	}
}

unsigned long long getNetReceivedBytes() {
    std::ifstream file("/proc/net/dev");
	const std::string interface("enp0s31f6");
    std::string line;
    while (std::getline(file, line)) {
        if (line.find(interface) != std::string::npos) {
            std::istringstream iss(line);
            std::string temp;
            iss >> temp;
            unsigned long long bytes;
            iss >> bytes;
            return bytes;
        }
    }
    return 0;
};

// Minimal PLY loader for background
// WARNING: Assumes binary_little_endian format and specific property order.
void sibr::GaussianView::loadBackground(const std::string& ply_path)
{
	SIBR_LOG << "Loading background PLY: " << ply_path << std::endl;
    std::ifstream file(ply_path, std::ios::binary);
    if (!file.is_open()) {
        SIBR_ERR << "ERROR: Could not open PLY file: " << ply_path;
        return;
    }

    std::string line;
    int vertex_count = 0;
    std::string format;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        ss >> token;
        if (token == "format") {
            ss >> format;
        } else if (token == "element") {
            std::string type;
            ss >> type;
            if (type == "vertex") {
                ss >> vertex_count;
            }
        } else if (token == "end_header") {
            break;
        }
    }

    if (format != "binary_little_endian" || vertex_count == 0) {
        SIBR_ERR << "ERROR: Unsupported PLY format (" << format << ") or zero vertices for background.";
        file.close();
        return;
    }

	background_count = vertex_count;
	float background_scale = bg_scales[current_video_item];
    SIBR_LOG << "Background vertex count: " << background_count;

    // Allocate CPU vectors
    std::vector<Pos> pos_vec(background_count);
    std::vector<Rot> rot_vec(background_count);
    std::vector<Scale> scale_vec(background_count);
    std::vector<float> opacity_vec(background_count);
    std::vector<SHs<3>> shs_vec(background_count);

    // Read raw data
    std::vector<PlyGaussian> ply_data(background_count);
    file.read(reinterpret_cast<char*>(ply_data.data()), background_count * sizeof(PlyGaussian));
    file.close();

    // Convert to separate attribute vectors
	// This assumes the PLY stores SHs up to degree 3
    for (int i = 0; i < background_count; ++i) {
        pos_vec[i] = Pos(ply_data[i].x * background_scale, ply_data[i].y * background_scale, ply_data[i].z * background_scale);

        // Normalize rotation quaternion
        double norm = std::sqrt(ply_data[i].rot_0 * ply_data[i].rot_0 +
                                ply_data[i].rot_1 * ply_data[i].rot_1 +
                                ply_data[i].rot_2 * ply_data[i].rot_2 +
                                ply_data[i].rot_3 * ply_data[i].rot_3);
        rot_vec[i].rot[0] = static_cast<float>(ply_data[i].rot_0 / norm);
        rot_vec[i].rot[1] = static_cast<float>(ply_data[i].rot_1 / norm);
        rot_vec[i].rot[2] = static_cast<float>(ply_data[i].rot_2 / norm);
        rot_vec[i].rot[3] = static_cast<float>(ply_data[i].rot_3 / norm);

        // Apply exp to scale
        scale_vec[i].scale[0] = std::exp(ply_data[i].scale_0) * background_scale;;
        scale_vec[i].scale[1] = std::exp(ply_data[i].scale_1) * background_scale;;
        scale_vec[i].scale[2] = std::exp(ply_data[i].scale_2) * background_scale;;

        // Apply sigmoid to opacity
        opacity_vec[i] = sigmoid(ply_data[i].opacity);

        // Copy SHs
        shs_vec[i].shs[0] = ply_data[i].f_dc_0;
        shs_vec[i].shs[1] = ply_data[i].f_dc_1;
        shs_vec[i].shs[2] = ply_data[i].f_dc_2;
        memcpy(&shs_vec[i].shs[3], ply_data[i].f_rest, 45 * sizeof(float));
    }

    // Allocate GPU memory for background
	const int shs_size_allocated = sizeof(SHs<3>);
    CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&background_pos_cuda, sizeof(Pos) * background_count));
    CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&background_rot_cuda, sizeof(Rot) * background_count));
    CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&background_scale_cuda, sizeof(Scale) * background_count));
    CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&background_opacity_cuda, sizeof(float) * background_count));
    CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&background_shs_cuda, shs_size_allocated * background_count));

    // Copy to GPU
    CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(background_pos_cuda, pos_vec.data(), sizeof(Pos) * background_count, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(background_rot_cuda, rot_vec.data(), sizeof(Rot) * background_count, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(background_scale_cuda, scale_vec.data(), sizeof(Scale) * background_count, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(background_opacity_cuda, opacity_vec.data(), sizeof(float) * background_count, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(background_shs_cuda, shs_vec.data(), shs_size_allocated * background_count, cudaMemcpyHostToDevice));

	SIBR_LOG << "Background PLY loaded successfully.";
}


namespace sibr
{
	static float s_fps = 30.0f; // Frames per second for video playback
	// A simple copy renderer class. Much like the original, but this one
	// reads from a buffer instead of a texture and blits the result to
	// a render target. 
	class BufferCopyRenderer
	{

	public:

		BufferCopyRenderer()
		{
			_shader.init("CopyShader",
				sibr::loadFile(sibr::getShadersDirectory("gaussian") + "/copy.vert"),
				sibr::loadFile(sibr::getShadersDirectory("gaussian") + "/copy.frag"));

			_flip.init(_shader, "flip");
			_width.init(_shader, "width");
			_height.init(_shader, "height");
		}

		void process(uint bufferID, IRenderTarget& dst, int width, int height, bool disableTest = true)
		{
			if (disableTest)
				glDisable(GL_DEPTH_TEST);
			else
				glEnable(GL_DEPTH_TEST);

			_shader.begin();
			_flip.send();
			_width.send();
			_height.send();

			dst.clear();
			dst.bind();

			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bufferID);

			sibr::RenderUtility::renderScreenQuad();

			dst.unbind();
			_shader.end();
		}

		/** \return option to flip the texture when copying. */
		bool& flip() { return _flip.get(); }
		int& width() { return _width.get(); }
		int& height() { return _height.get(); }

	private:

		GLShader			_shader; 
		GLuniform<bool>		_flip = false; ///< Flip the texture when copying.
		GLuniform<int>		_width = 1000;
		GLuniform<int>		_height = 800;
	};
}

std::function<char* (size_t N)> resizeFunctional(void** ptr, size_t& S) {
	auto lambda = [ptr, &S](size_t N) {
		if (N > S)
		{
			if (*ptr)
				CUDA_SAFE_CALL(cudaFree(*ptr));
			CUDA_SAFE_CALL(cudaMalloc(ptr, 2 * N));
			S = 2 * N;
		}
		return reinterpret_cast<char*>(*ptr);
	};

	return lambda;
}

sibr::GaussianView::GaussianView(const sibr::BasicIBRScene::Ptr & ibrScene, uint render_w, uint render_h, const char* file, bool* messageRead, bool white_bg, bool useInterop, int device) :
	_scene(ibrScene),
	_dontshow(messageRead),
	sibr::ViewBase(render_w, render_h)
{
	// Initialize frame tracker
	last_loaded_frame_id = -1;

	int num_devices;
	CUDA_SAFE_CALL_ALWAYS(cudaGetDeviceCount(&num_devices));
	_device = device;
	if (device >= num_devices)
	{
		if (num_devices == 0)
			SIBR_ERR << "No CUDA devices detected!";
		else
			SIBR_ERR << "Provided device index exceeds number of available CUDA devices!";
	}
	CUDA_SAFE_CALL_ALWAYS(cudaSetDevice(device));
	cudaDeviceProp prop;
	CUDA_SAFE_CALL_ALWAYS(cudaGetDeviceProperties(&prop, device));

	_pointbasedrenderer.reset(new PointBasedRenderer());
	_copyRenderer = new BufferCopyRenderer();
	_copyRenderer->flip() = true;
	_copyRenderer->width() = _resolution.x();
	_copyRenderer->height() = _resolution.y();

	std::vector<uint> imgs_ulr;
	const auto & cams = ibrScene->cameras()->inputCameras();
	for(size_t cid = 0; cid < cams.size(); ++cid) {
		if(cams[cid]->isActive()) {
			imgs_ulr.push_back(uint(cid));
		}
	}
	_scene->cameras()->debugFlagCameraAsUsed(imgs_ulr);


	// init folder
	folder = std::string(video_path[current_video_item]);
	_sh_degree = video_sh[current_video_item];
	num_att_index = (14 + (3 * (_sh_degree + 1) * (_sh_degree + 1))) + 3;

	// multi-frame setting
	global_png_vector.resize(num_att_index);

	// init download and ready to 0
	memset(downloaded_array, 0, sizeof(downloaded_array));
	memset(ready_array, 0, sizeof(ready_array));

	// std::string folder = "http://10.15.89.67:10000/video_new/";
	// std::string group_json_path = "http://10.15.89.67:10000/group_info.json";
	std::string group_json_path = folder + "group_info.json";
	picojson::object group_obj = fetchJsonObj(group_json_path);
	size_t num_groups = group_obj.size();
	// for (const auto& kv : group_obj) {
	// 	picojson::object innerObj = kv.second.get<picojson::object>();
    //     picojson::array frame_index = innerObj["frame_index"].get<picojson::array>();
    //     group_frame_index.push_back(std::make_pair((int)frame_index[0].get<double>(), (int)frame_index[1].get<double>()));
    //     picojson::array name_index = innerObj["name_index"].get<picojson::array>();
    //     group_name_index.push_back(std::make_pair((int)name_index[0].get<double>(), (int)name_index[1].get<double>()));
	// }
	for (int i = 0; i < num_groups; i++) {
		picojson::object innerObj = group_obj[std::to_string(i)].get<picojson::object>();
		picojson::array frame_index = innerObj["frame_index"].get<picojson::array>();
		group_frame_index.push_back(std::make_pair((int)frame_index[0].get<double>(), (int)frame_index[1].get<double>()));
		picojson::array name_index = innerObj["name_index"].get<picojson::array>();
		group_name_index.push_back(std::make_pair((int)name_index[0].get<double>(), (int)name_index[1].get<double>()));
	}
	download_cache_size = std::min(download_cache_size, (int)num_groups);

	// get last frame index
	sequences_length = group_frame_index[group_frame_index.size() - 1].second;
	std::cout << "Total frames: " << sequences_length << std::endl;
	// init global_png_vector for each attribute
	for (int i = 0; i < num_att_index; i++) {
		global_png_vector[i].resize(sequences_length + 1);
	}

	auto start1 = std::chrono::high_resolution_clock::now();

	std::vector<std::future<bool>> futures;
	int initial_group_index = 0;
	int download_start_index = group_frame_index[initial_group_index].first;
	int download_end_index = group_frame_index[initial_group_index].second;
	for (int att_index = 0; att_index < num_att_index; att_index ++) {
		std::string videopath = folder + "group" + std::to_string(initial_group_index) + "/" + std::to_string(att_index) + ".mp4";
        futures.push_back(std::async(std::launch::async, getAllFramesNew, videopath, download_start_index, std::ref(global_png_vector[att_index])));
	}

	for (auto& f : futures) {
        bool success = f.get(); // This will wait for the thread to finish
        if (!success) {
            std::cerr << "Failed to process some videos" << std::endl;
        }
    }
	// downloaded_frames = global_png_vector[0].size();
	num_frames = download_end_index - download_start_index + 1;
	downloaded_frames = download_end_index;
	// set downloaded from download_start_index to download_end_index
	std::fill(downloaded_array + download_start_index, downloaded_array + download_end_index + 1, 1);

	auto end1 = std::chrono::high_resolution_clock::now();
	double elapsed_seconds1 = std::chrono::duration<double>(end1 - start1).count();
	// std::cout << "OpenCV read 100 frames: " << elapsed_seconds1 << " seconds" << std::endl;

	// read the json info
	std::string minmax_json_path = folder + "viewer_min_max.json";
	minmax_obj = fetchJsonObj(minmax_json_path);

	// start timer

	const int shs_size_allocated = sizeof(SHs<3>);
	for (int i = 0; i < GPU_RING_BUFFER_SLOTS; ++i)
	{
		CUDA_SAFE_CALL_ALWAYS(cudaStreamCreate(&data_streams[i]));
		CUDA_SAFE_CALL_ALWAYS(cudaEventCreate(&data_events[i]));

		CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&gpu_ring_buffer[i].pos_cuda, sizeof(Pos) * MAX_GAUSSIANS_PER_FRAME));
		CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&gpu_ring_buffer[i].rot_cuda, sizeof(Rot) * MAX_GAUSSIANS_PER_FRAME));
		CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&gpu_ring_buffer[i].scale_cuda, sizeof(Scale) * MAX_GAUSSIANS_PER_FRAME));
		CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&gpu_ring_buffer[i].opacity_cuda, sizeof(float) * MAX_GAUSSIANS_PER_FRAME));
		CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&gpu_ring_buffer[i].shs_cuda, shs_size_allocated * MAX_GAUSSIANS_PER_FRAME));
		CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&gpu_ring_buffer[i].rect_cuda, 2 * MAX_GAUSSIANS_PER_FRAME * sizeof(int)));

		CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&gpu_ring_buffer[i].raw_attributes_cuda, MAX_ATTRIBUTES * MAX_IMAGE_PIXELS * sizeof(uint8_t)));
        CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&gpu_ring_buffer[i].minmax_values_cuda, MAX_ATTRIBUTES * 2 * sizeof(float)));
	}
	
    // --- Load background ---
	_background_ply_path = bg_paths[current_video_item];
    if (!_background_ply_path.empty()) {
        loadBackground(_background_ply_path);
    }

    // --- Allocate combined buffers ---
    combined_buffer_allocated_count = background_count + MAX_GAUSSIANS_PER_FRAME;
    SIBR_LOG << "Allocating combined buffers for " << combined_buffer_allocated_count << " Gaussians.";
    CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&combined_pos_cuda, sizeof(Pos) * combined_buffer_allocated_count));
    CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&combined_rot_cuda, sizeof(Rot) * combined_buffer_allocated_count));
    CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&combined_scale_cuda, sizeof(Scale) * combined_buffer_allocated_count));
    CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&combined_opacity_cuda, sizeof(float) * combined_buffer_allocated_count));
    CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&combined_shs_cuda, shs_size_allocated * combined_buffer_allocated_count));
    CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&combined_rect_cuda, 2 * combined_buffer_allocated_count * sizeof(int)));

    // --- Create combine stream ---
    CUDA_SAFE_CALL_ALWAYS(cudaStreamCreate(&combine_stream));
    CUDA_SAFE_CALL_ALWAYS(cudaEventCreate(&combine_event));

	auto start = std::chrono::high_resolution_clock::now();
	// std::cout << "preload index start: " << group_frame_index[0].first << std::endl;
	// std::cout << "preload index end: " << group_frame_index[0].second << std::endl;
	// for (int i = group_frame_index[0].first; i <= group_frame_index[0].second; i+=frame_step)
	// std::cout << "group frame index" << group_frame_index[0].second << std::endl;
	for (int i = download_start_index; i < download_start_index + std::min(ready_cache_size, download_end_index+1); i+=frame_step)
	{	
		std::cout << "Preloading frame " << i << std::endl;
		loadVideo_func(i);
	}
	auto end = std::chrono::high_resolution_clock::now();
	double elapsed_seconds = std::chrono::duration<double>(end - start).count();
	std::cout << "Elapsed time: " << elapsed_seconds << " seconds" << std::endl;
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&view_cuda, sizeof(sibr::Matrix4f)));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&proj_cuda, sizeof(sibr::Matrix4f)));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&proj_inv_cuda, sizeof(sibr::Matrix4f)));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&cam_pos_cuda, 3 * sizeof(float)));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&background_cuda, 3 * sizeof(float)));

    // The previous allocation was (num_tiles / 8) bytes, which causes overflow if the rasterizer writes 
    // a list of indices (4 bytes per tile).
	int max_num_tiles = ((_resolution.x() + 15) / 16) * ((_resolution.y() + 15) / 16);
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&visibility_mask_cuda, (((_resolution.x() + 15) / 16) * ((_resolution.y() + 15) / 16) + 31) / 32 * 4));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc((void**)&visibility_mask_sum_cuda, ((_resolution.x() + 15) / 16 + 1) * ((_resolution.y() + 15) / 16 + 1) * 4));
	
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc(&image_cuda_hier[0], (_resolution.x() / 2) * (_resolution.y() / 2) * 3 * sizeof(float)));
	CUDA_SAFE_CALL_ALWAYS(cudaMalloc(&image_cuda_hier[1], (_resolution.x() / 2) * (_resolution.y() / 2) * 3 * sizeof(float)));

	// float bg[3] = { white_bg ? 1.f : 0.f, white_bg ? 1.f : 0.f, white_bg ? 1.f : 0.f };
	float bg[3] = { 0.f, 0.f, 0.f };
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(background_cuda, bg, 3 * sizeof(float), cudaMemcpyHostToDevice));
	
	std::cout << "Waiting for frame 0 to be ready..." << std::endl;
    int first_slot = 0 % GPU_RING_BUFFER_SLOTS;
    CUDA_SAFE_CALL_ALWAYS(cudaEventSynchronize(data_events[first_slot]));
    std::cout << "Frame 0 is ready." << std::endl;

    // Set render pointers to the combined buffers
    // The initial 'count' will be set in the first onUpdate call.
	count = 0; 
	pos_cuda = combined_pos_cuda;
	rot_cuda = combined_rot_cuda;
	scale_cuda = combined_scale_cuda;
	opacity_cuda = combined_opacity_cuda;
	shs_cuda = combined_shs_cuda;
	rect_cuda = combined_rect_cuda;

	_gaussianRenderer = new GaussianSurfaceRenderer();
	createImageBuffer();
	geomBufferFunc = resizeFunctional(&geomPtr, allocdGeom);
	binningBufferFunc = resizeFunctional(&binningPtr, allocdBinning);
	imgBufferFunc = resizeFunctional(&imgPtr, allocdImg);

	parseJSON();
}

void sibr::GaussianView::parseJSON()
{
	std::string json_path = _scene->data()->configPath();
	std::ifstream json_file(json_path, std::ios::in);

	splatting_settings = CudaRasterizer::SplattingSettings();

	// return if no config file is found - use default parameters (Vanilla 3DGS)
 	if (json_file.fail()) return;
	nlohmann::json js = nlohmann::json::parse(json_file);

	// get settings from config
	splatting_settings = js.get<CudaRasterizer::SplattingSettings>();

	if (splatting_settings.foveated_rendering)
		SIBR_LOG << "Using Foveated Rendering" << std::endl;
	else
		SIBR_LOG << "Not Using Foveated Rendering" << std::endl;

	// sanity checks
	if (CudaRasterizer::isInvalidSortMode(splatting_settings.sort_settings.sort_mode))
	{
		SIBR_LOG << "Invalid Sort Mode in " << json_path << " ("<< splatting_settings.sort_settings.sort_mode << "): continuing with default" << std::endl;
		splatting_settings.sort_settings.sort_mode = CudaRasterizer::SortMode::GLOBAL;
	}
	if (CudaRasterizer::isInvalidSortOrder(splatting_settings.sort_settings.sort_order))
	{
		SIBR_LOG << "Invalid Sort Order in " << json_path << " ("<< splatting_settings.sort_settings.sort_order << "): continuing with default" << std::endl;
		splatting_settings.sort_settings.sort_order = CudaRasterizer::GlobalSortOrder::VIEWSPACE_Z;
	}
	if (splatting_settings.sort_settings.hasModifiableWindowSize())
	{
		auto sort_mode = splatting_settings.sort_settings.sort_mode;
		auto test_function = [&](std::vector<int> vec, const char* what, int default_value, int& variable)
		{
			if (std::find(vec.begin(), vec.end(), variable) == vec.end())
			{
				SIBR_LOG << "Invalid " << what << " Size in " << json_path << " ("<< variable << "): continuing with default" << std::endl;
				variable = default_value;
			}
		};
	if (sort_mode == CudaRasterizer::SortMode::HIERARCHICAL)
		{
			test_function(CudaRasterizer::per_pixel_queue_sizes_hier, "Per-Pixel Queue", 4, splatting_settings.sort_settings.queue_sizes.per_pixel);
			test_function(CudaRasterizer::twobytwo_tile_queue_sizes, "2x2-Tile Queue", 8, splatting_settings.sort_settings.queue_sizes.tile_2x2);
		}
		if (sort_mode == CudaRasterizer::SortMode::PER_PIXEL_KBUFFER)
		{
			test_function(CudaRasterizer::per_pixel_queue_sizes, "Per-Pixel Queue", 1, splatting_settings.sort_settings.queue_sizes.per_pixel);
		}
	}
}

void sibr::GaussianView::setResolution(const Vector2i &size)
{
	if (size != getResolution())
	{
		// SIBR_LOG << "Set resolution => " << size << std::endl;
		ViewBase::setResolution(size);
		destroyImageBuffer();
		createImageBuffer();

		_copyRenderer->width() = _resolution.x();
		_copyRenderer->height() = _resolution.y();
	}
}

void sibr::GaussianView::createImageBuffer()
{
	try {
		// Create GL buffer ready for CUDA/GL interop
		glCreateBuffers(1, &imageBuffer);
		glNamedBufferStorage(imageBuffer, _resolution.x() * _resolution.y() * 3 * sizeof(float), nullptr, GL_DYNAMIC_STORAGE_BIT);

		if (_use_interop)
		{
			if (cudaPeekAtLastError() != cudaSuccess)
			{
				SIBR_ERR << "A CUDA error occurred in setup:" << cudaGetErrorString(cudaGetLastError()) << ". Please rerun in Debug to find the exact line!";
			}
			cudaGraphicsGLRegisterBuffer(&imageBufferCuda, imageBuffer, cudaGraphicsRegisterFlagsWriteDiscard);
			_use_interop &= (cudaGetLastError() == cudaSuccess);
		}
		if (!_use_interop)
		{
			fallback_bytes.resize(_resolution.x() * _resolution.y() * 3 * sizeof(float));
			cudaMalloc(&fallbackBufferCuda, fallback_bytes.size());
			_interop_failed = true;
		}

		// geomBufferFunc = resizeFunctional(&geomPtr, allocdGeom);
		// binningBufferFunc = resizeFunctional(&binningPtr, allocdBinning);
		// imgBufferFunc = resizeFunctional(&imgPtr, allocdImg);

		// // Check system resources before creating threads
		// if (!checkSystemResources()) {
		// 	throw std::runtime_error("System resource check failed: Insufficient resources to continue.");
		// }
		
		// Safely create/restart threads with proper exception handling
		std::lock_guard<std::mutex> thread_lock(_thread_management_mutex);
		try {
			// Check if threads are already running and stop them properly
			if (download_thread_.joinable()) {
				frame_changed = true; // Signal threads to stop
				cv_download.notify_all();
				download_thread_.join();
			}
			if (ready_thread_.joinable()) {
				cv_ready.notify_all();
				ready_thread_.join();
			}
			
			// Reset frame_changed flag
			frame_changed = false;
			
			// Create new threads
			download_thread_ = std::thread(&sibr::GaussianView::download_func, this);
			
			{
				std::lock_guard<std::mutex> lock(mtx_ready);
				for (int group_index = 1; group_index < download_cache_size; group_index++) {
					need_download_q.push(group_index);
				}
			}
			cv_download.notify_one();

			ready_thread_ = std::thread(&sibr::GaussianView::readyVideo_func, this);
			
			// Mark threads as successfully initialized
			_threads_initialized = true;
			
		} catch (const std::system_error& e) {
			SIBR_ERR << "Failed to create worker threads: " << e.what() << ". Code: " << e.code() << std::endl;
			// Continue without threads if creation fails
		} catch (const std::exception& e) {
			SIBR_ERR << "Exception during thread creation: " << e.what() << std::endl;
		}

		// set up time recoder for stable play speed
		frameDuration = std::chrono::milliseconds(33);
		lastUpdateTimestamp = std::chrono::high_resolution_clock::now();

		// set up time recoder for memory read
		MemframeDuration = std::chrono::milliseconds(1000);
		MemlastUpdateTimestamp = std::chrono::high_resolution_clock::now();

		// read network
		last_total_bytes = getNetReceivedBytes();

	// std::cout << "the number of 0 gaussian" << count << std::endl;
	} catch (const std::exception& e) {
		SIBR_ERR << "Exception in createImageBuffer: " << e.what() << std::endl;
		throw; // Re-throw to allow proper error handling upstream
	}
}

bool sibr::GaussianView::checkSystemResources()
{
	try {
		// Check available system threads
		unsigned int hardware_threads = std::thread::hardware_concurrency();
		if (hardware_threads == 0) {
			SIBR_LOG << "Warning: Cannot determine number of hardware threads" << std::endl;
			return true; // Continue anyway
		}
		
		SIBR_LOG << "System has " << hardware_threads << " hardware threads available" << std::endl;
		
		// Basic memory check using sysinfo (Linux)
		#ifdef __linux__
			struct sysinfo info;
			if (sysinfo(&info) == 0) {
				unsigned long available_ram = info.freeram * info.mem_unit;
				unsigned long total_ram = info.totalram * info.mem_unit;
				SIBR_LOG << "Available RAM: " << available_ram / (1024*1024) << " MB / " 
				          << total_ram / (1024*1024) << " MB total" << std::endl;
				          
				// Check if we have at least 1GB available
				if (available_ram < 1024*1024*1024) {
					SIBR_ERR << "Warning: Low memory available (< 1GB). This may cause issues." << std::endl;
				}
			}
		#endif
		
		return true;
		
	} catch (const std::exception& e) {
		SIBR_ERR << "Exception during system resource check: " << e.what() << std::endl;
		return false;
	}
}

// download gaussian
void sibr::GaussianView::download_func() {

	int group_index;
	while (frame_changed == false) {
		{
			std::unique_lock<std::mutex> lock(mtx_download);
			cv_download.wait(lock, [this] { return !need_download_q.empty()|| frame_changed; });
			if (frame_changed) { //Check flag and exit loop if woken up to stop
				break;
			}

			if (need_download_q.empty()) { 
				continue;
			}
			group_index = need_download_q.front();
			// if (i > frame_id + download_cache_size) {
			// 	continue;
			// }
			need_download_q.pop();
		}
		{
			int download_start_index = group_frame_index[group_index].first;
			int download_end_index = group_frame_index[group_index].second;
			if (downloaded_array[download_start_index] == 0) {
				auto start = std::chrono::high_resolution_clock::now();
				std::cout << "Do not find cache" << std::endl;
				std::vector<std::future<bool>> thread_futures;
				for (int att_index = 0; att_index < num_att_index; att_index ++) {
					std::string videopath = folder + "group" + std::to_string(group_index) + "/" + std::to_string(att_index) + ".mp4";
					thread_futures.push_back(std::async(std::launch::async, getAllFramesNew, videopath, download_start_index, std::ref(global_png_vector[att_index])));
				}

				for (auto& f : thread_futures) {
					bool success = f.get(); // This will wait for the thread to finish
					if (!success) {
						std::cerr << "Failed to process some videos" << std::endl;
					}
				}
				auto end = std::chrono::high_resolution_clock::now();
				double elapsed_seconds = std::chrono::duration<double>(end - start).count();
				std::cout << "Download Elapsed time: " << elapsed_seconds << " seconds" << std::endl;
			} else {
				std::cout << "Cached group: " << group_index << std::endl;
				std::cout << "Cached frame index: " << download_start_index << " " << download_end_index << std::endl;
			}
			if (download_end_index > downloaded_frames) {
				downloaded_frames = download_end_index;
			}
			std::fill(downloaded_array + download_start_index, downloaded_array + download_end_index + 1, 1);
			std::lock_guard<std::mutex> lock(mtx_download);

		}
	}
}

void sibr::GaussianView::destroyImageBuffer()
{
	try {
		// Signal threads to stop and clean up properly
		frame_changed = true;
		
		// Wake up threads and wait for them to finish
		cv_download.notify_all();
		cv_ready.notify_all();
		
		// Join threads with timeout to avoid hanging
		if (download_thread_.joinable()) {
			try {
				download_thread_.join();
			} catch (const std::exception& e) {
				SIBR_ERR << "Exception joining download thread: " << e.what() << std::endl;
				// If join fails, detach to prevent termination
				download_thread_.detach();
			}
		}
		
		if (ready_thread_.joinable()) {
			try {
				ready_thread_.join();
			} catch (const std::exception& e) {
				SIBR_ERR << "Exception joining ready thread: " << e.what() << std::endl;
				// If join fails, detach to prevent termination
				ready_thread_.detach();
			}
		}
		
		// Clean up GPU resources
		if (_use_interop)
		{
			cudaGraphicsUnregisterResource(imageBufferCuda);
		}
		else
		{
			cudaFree(fallbackBufferCuda);
		}
		glDeleteBuffers(1, &imageBuffer);
		imageBuffer = 0;
		
	} catch (const std::exception& e) {
		SIBR_ERR << "Exception in destroyImageBuffer: " << e.what() << std::endl;
		// Continue cleanup even if exceptions occur
		try {
			glDeleteBuffers(1, &imageBuffer);
			imageBuffer = 0;
		} catch (...) {
			// Ignore further exceptions during emergency cleanup
		}
	}
}

void sibr::GaussianView::setScene(const sibr::BasicIBRScene::Ptr & newScene)
{
	_scene = newScene;

	// Tell the scene we are a priori using all active cameras.
	std::vector<uint> imgs_ulr;
	const auto & cams = newScene->cameras()->inputCameras();
	for (size_t cid = 0; cid < cams.size(); ++cid) {
		if (cams[cid]->isActive()) {
			imgs_ulr.push_back(uint(cid));
		}
	}
	_scene->cameras()->debugFlagCameraAsUsed(imgs_ulr);
}

void sibr::GaussianView::onRenderIBR(sibr::IRenderTarget & dst, const sibr::Camera & eye)
{
	if (count <= 0) {
		// Clear the destination to black to avoid showing the previous frame's image.
        dst.clear(); // Clear screen
        return;
	}

    // Wait for the combine stream (if data is being composited)
    if (count > 0) {
        // Wait on the default stream (stream 0) for the combine_event to complete
        CUDA_SAFE_CALL(cudaStreamWaitEvent(0, combine_event, 0));
    }

	if (currMode == "Ellipsoids")
	{
		_gaussianRenderer->process(count, *gData, eye, dst, 0.2f);
	}
	else if (currMode == "Initial Points")
	{
		_pointbasedrenderer->process(_scene->proxies()->proxy(), eye, dst);
	}
	else
	{


		auto forward = [&](const sibr::Camera& eye, float* image_cuda_curr, int x, int y, bool mask = false) {
			// Convert view and projection to target coordinate system
			auto view_mat = eye.view();
			auto proj_mat = eye.viewproj();
			view_mat.row(1) *= -1;
			view_mat.row(2) *= -1;
			proj_mat.row(1) *= -1;

			// Compute additional view parameters
			float tan_fovy = tan(eye.fovy() * 0.5f);
			float tan_fovx = tan_fovy * eye.aspect();

			auto proj_inv_mat = sibr::Matrix4f(proj_mat.inverse());

			// Copy frame-dependent data to GPU
			CUDA_SAFE_CALL(cudaMemcpy(view_cuda, view_mat.data(), sizeof(sibr::Matrix4f), cudaMemcpyHostToDevice));
			CUDA_SAFE_CALL(cudaMemcpy(proj_cuda, proj_mat.data(), sizeof(sibr::Matrix4f), cudaMemcpyHostToDevice));
			CUDA_SAFE_CALL(cudaMemcpy(proj_inv_cuda, proj_inv_mat.data(), sizeof(sibr::Matrix4f), cudaMemcpyHostToDevice));
			CUDA_SAFE_CALL(cudaMemcpy(cam_pos_cuda, &eye.position(), sizeof(float) * 3, cudaMemcpyHostToDevice));
			if (mask)
			{
				// auto start = std::chrono::steady_clock::now();
				CUDA_SAFE_CALL(cudaMemcpy(visibility_mask_cuda, eye.visibilityMask().first, (((_resolution.x() + 15) / 16) * ((_resolution.y() + 15) / 16) + 31) / 32 * 4, cudaMemcpyHostToDevice));
				CUDA_SAFE_CALL(cudaMemcpy(visibility_mask_sum_cuda, eye.visibilityMask().second, ((_resolution.x() + 15) / 16 + 1) * ((_resolution.y() + 15) / 16 + 1) * 4, cudaMemcpyHostToDevice));
			}

			// Rasterize
			CudaRasterizer::Rasterizer::forward(
				geomBufferFunc,
				binningBufferFunc,
				imgBufferFunc,
				count, _sh_degree, 16,
				background_cuda,
				x, y,
				splatting_settings,
				debugMode,
				pos_cuda,
				shs_cuda,
				nullptr,
				opacity_cuda,
				scale_cuda,
				_scalingModifier,
				rot_cuda,
				nullptr,
				view_cuda,
				proj_cuda,
				proj_inv_cuda,
				cam_pos_cuda,
				tan_fovx,
				tan_fovy,
				false,
				image_cuda_curr,
				nullptr,
				false,
				mask ? visibility_mask_cuda : nullptr,
				mask ? visibility_mask_sum_cuda : nullptr
			);
		};

		float* image_cuda = nullptr;
		if (!_interop_failed)
		{
			// Map OpenGL buffer resource for use with CUDA
			size_t bytes;
			CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &imageBufferCuda));
			CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void**)&image_cuda, &bytes, imageBufferCuda));
		}
		else
		{
			image_cuda = fallbackBufferCuda;
		}

		int w = _resolution.x();
		int h = _resolution.y();

		Camera eye2 = eye;
		if (eye2.isSym()) {
			float fovv = eye.fovy();
			float fovh = fovv * eye2.aspect();
			eye2.setAllFov({ -fovh / 2, fovh / 2, -fovv / 2, fovv / 2 });
		}

		// Low-res
		forward(eye2, image_cuda_hier[0], w / 2, h / 2, !eye.isSym());

		// High-res
		auto fov = eye2.allFov();
		// eye2.fovy(atan(tan((fov.w() - fov.z()) / 2) * 0.5f) * 2);
		eye2.fovy(atan(tan(fov.w()) * 0.5f) - atan(tan(fov.z()) * 0.5f));
		eye2.setAllFov({atan(tan(fov.x()) * 0.5f), atan(tan(fov.y()) * 0.5f), atan(tan(fov.z()) * 0.5f), atan(tan(fov.w()) * 0.5f)});
		// fov = eye2.allFov();
		forward(eye2, image_cuda_hier[1], w / 2, h / 2);


		// Upsample
		{
			NppiSize srcSize = { w / 2, h / 2 };
			const float* pSrc[] = {
				image_cuda_hier[0],
				image_cuda_hier[0] + srcSize.width * srcSize.height,
				image_cuda_hier[0] + 2 * srcSize.width * srcSize.height
			};
			int srcStep = srcSize.width * sizeof(float);
			NppiRect srcRect = { 0, 0, srcSize.width, srcSize.height };
			NppiSize dstSize = { w, h };
			float* pDst[] = {
				image_cuda,
				image_cuda + w * h,
				image_cuda + 2 * w * h
			};
			int dstStep = w * sizeof(float);
			NppiRect dstRect = { 0, 0, w / 2 * 2, h / 2 * 2 };
			auto status = nppiResize_32f_P3R(
				pSrc, srcStep, srcSize, srcRect,
				pDst, dstStep, dstSize, dstRect,
				NPPI_INTER_CUBIC
				// NPPI_INTER_LANCZOS
			);
			if (status != NPP_SUCCESS)
			{
				SIBR_ERR << "NPP error: " << status << std::endl;
			}
		}

		// Move high-res image
		{
			CudaRasterizer::blend(
				image_cuda_hier[1],
				w / 2, h / 2,
				image_cuda,
				w, h,
				w / (tan(fov.y()) - tan(fov.x())) * tan(-fov.x()) / 2 + 0.5,
				h / (tan(fov.w()) - tan(fov.z())) * tan(fov.w()) / 2 + 0.5,
				0.1f
			);
		}

		if (!_interop_failed)
		{
			// Unmap OpenGL resource for use with OpenGL
			CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &imageBufferCuda));
		}
		else
		{
			CUDA_SAFE_CALL(cudaMemcpy(fallback_bytes.data(), fallbackBufferCuda, fallback_bytes.size(), cudaMemcpyDeviceToHost));
			glNamedBufferSubData(imageBuffer, 0, fallback_bytes.size(), fallback_bytes.data());
		}
		// Copy image contents to framebuffer
		_copyRenderer->process(imageBuffer, dst, _resolution.x(), _resolution.y());
	}

	if (cudaPeekAtLastError() != cudaSuccess)
	{
		SIBR_ERR << "A CUDA error occurred during rendering:" << cudaGetErrorString(cudaGetLastError()) << ". Please rerun in Debug to find the exact line!";
	}
}

void sibr::GaussianView::onUpdate(Input & input)
{
	// Update frame duration based on FPS slider from GUI
	frameDuration = std::chrono::milliseconds((int)(1000.0f / s_fps));

	auto now = std::chrono::high_resolution_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastUpdateTimestamp);

	int current_frame_id = 0;
	{
		std::lock_guard<std::mutex> lock(mtx_frame_id);
		current_frame_id = frame_id;
	}

	if (_multi_view_play.load()) {
		if (elapsed > frameDuration) {

			int next_frame_id = -1;

			if (current_frame_id == sequences_length) {
				std::cout << "[onUpdate] Playback reached end. Looping back to 0." << std::endl;
				next_frame_id = 0;

			} else if (ready_array[current_frame_id + 1] == 1) { // 2. Not at end, check next frame
				next_frame_id = current_frame_id + 1;
				std::cout << "[onUpdate] Advancing from " << current_frame_id << " to " << current_frame_id + 1 << std::endl;
			} else {
				std::cout << "[onUpdate] Waiting for frame after " << current_frame_id << " to become ready..." << std::endl;
				std::this_thread::sleep_for(std::chrono::milliseconds(10)); // Shorter sleep
			}

			// 4. If we have a valid next frame, apply the change
			if (next_frame_id != -1)
			{
				
				int frame_to_free = next_frame_id - 1;
				if(frame_to_free >= 0) {
					ready_array[frame_to_free] = 0;
				}

				// Update frame_id
				{ 
					std::lock_guard<std::mutex> lock(mtx_frame_id);
					frame_id = next_frame_id;
					current_frame_id = frame_id; // Update local copy
				}
				lastUpdateTimestamp = now;
				

				// Download frame in buffer
				if ((current_frame_id-1) % num_frames == 0) { 
					int total_groups = group_frame_index.size();
					if (total_groups > 0) { 
						int next_group_to_download_logical_index = (current_frame_id / num_frames) + download_cache_size;
						int group_to_download_actual_index = next_group_to_download_logical_index % total_groups;
						
						std::cout << "[onUpdate] Requesting group " << group_to_download_actual_index << " to be downloaded." << std::endl;
						need_download_q.push(group_to_download_actual_index);
						cv_download.notify_one();
					}
				}

				// Ready frame in buffer
				int total_frames = sequences_length + 1;
				int frame_to_ready = (current_frame_id + ready_cache_size - 1) % total_frames;
				std::cout << "[onUpdate] Requesting frame " << frame_to_ready << " to be readied." << std::endl;
				need_ready_q.push(frame_to_ready);
				cv_ready.notify_one();
			}
		}
	}


	if (current_frame_id < sequences_length) {
		if (ready_array[current_frame_id] == 1 && current_frame_id != last_loaded_frame_id) {
            int slot = current_frame_id % GPU_RING_BUFFER_SLOTS;

            // This is the critical wait. If the frame isn't ready,
            // the render thread will pause here until the data is on the GPU.
            CUDA_SAFE_CALL_ALWAYS(cudaEventSynchronize(data_events[slot]));

            int dynamic_count = P_array[current_frame_id];

            if (dynamic_count > 0 || background_count > 0) {
                GpuFrameSlot& current_slot = gpu_ring_buffer[slot];
                
                // Destination pointers for combined buffers
                float* d_pos_dst = combined_pos_cuda;
                float* d_rot_dst = combined_rot_cuda;
                float* d_scale_dst = combined_scale_cuda;
                float* d_opacity_dst = combined_opacity_cuda;
                float* d_shs_dst = combined_shs_cuda;
                int* d_rect_dst = combined_rect_cuda;

                const int shs_size_allocated = sizeof(SHs<3>);
                size_t shs_floats_per_gaussian = shs_size_allocated / sizeof(float);

                // 1. Copy background (if it exists)
                if (background_count > 0) {
                    CUDA_SAFE_CALL(cudaMemcpyAsync(d_pos_dst, background_pos_cuda, sizeof(Pos) * background_count, cudaMemcpyDeviceToDevice, combine_stream));
                    d_pos_dst += background_count * 3; // 3 floats per Pos
                    CUDA_SAFE_CALL(cudaMemcpyAsync(d_rot_dst, background_rot_cuda, sizeof(Rot) * background_count, cudaMemcpyDeviceToDevice, combine_stream));
                    d_rot_dst += background_count * 4; // 4 floats per Rot
                    CUDA_SAFE_CALL(cudaMemcpyAsync(d_scale_dst, background_scale_cuda, sizeof(Scale) * background_count, cudaMemcpyDeviceToDevice, combine_stream));
                    d_scale_dst += background_count * 3; // 3 floats per Scale
                    CUDA_SAFE_CALL(cudaMemcpyAsync(d_opacity_dst, background_opacity_cuda, sizeof(float) * background_count, cudaMemcpyDeviceToDevice, combine_stream));
                    d_opacity_dst += background_count;
                    CUDA_SAFE_CALL(cudaMemcpyAsync(d_shs_dst, background_shs_cuda, shs_size_allocated * background_count, cudaMemcpyDeviceToDevice, combine_stream));
                    d_shs_dst += background_count * shs_floats_per_gaussian;
                    if (_fastCulling) {
                        // Clear rects for static background
                        CUDA_SAFE_CALL(cudaMemsetAsync(d_rect_dst, 0, 2 * sizeof(int) * background_count, combine_stream));
                        d_rect_dst += background_count * 2;
                    }
                }

                // 2. Copy dynamic frame data *after* background data
                if (dynamic_count > 0) {
                    CUDA_SAFE_CALL(cudaMemcpyAsync(d_pos_dst, current_slot.pos_cuda, sizeof(Pos) * dynamic_count, cudaMemcpyDeviceToDevice, combine_stream));
                    CUDA_SAFE_CALL(cudaMemcpyAsync(d_rot_dst, current_slot.rot_cuda, sizeof(Rot) * dynamic_count, cudaMemcpyDeviceToDevice, combine_stream));
                    CUDA_SAFE_CALL(cudaMemcpyAsync(d_scale_dst, current_slot.scale_cuda, sizeof(Scale) * dynamic_count, cudaMemcpyDeviceToDevice, combine_stream));
                    CUDA_SAFE_CALL(cudaMemcpyAsync(d_opacity_dst, current_slot.opacity_cuda, sizeof(float) * dynamic_count, cudaMemcpyDeviceToDevice, combine_stream));
                    CUDA_SAFE_CALL(cudaMemcpyAsync(d_shs_dst, current_slot.shs_cuda, shs_size_allocated * dynamic_count, cudaMemcpyDeviceToDevice, combine_stream));
                    if (_fastCulling) {
                        // Copy rects for dynamic part
                        CUDA_SAFE_CALL(cudaMemcpyAsync(d_rect_dst, current_slot.rect_cuda, 2 * sizeof(int) * dynamic_count, cudaMemcpyDeviceToDevice, combine_stream));
                    }
                }

                // Set render pointers (they already point to combined buffers, just set count)
				// count = background_count;
                count = background_count + dynamic_count;

				// Update tracker so we don't re-upload this frame
				last_loaded_frame_id = current_frame_id;

				// Record event when copies are done
                CUDA_SAFE_CALL(cudaEventRecord(combine_event, combine_stream));
            } 
		}
	} 
}

void sibr::GaussianView::onGUI()
{
	// Generate and update UI elements
	const std::string guiName = "3D Gaussians";
	if (ImGui::Begin(guiName.c_str())) 
	{
		if (ImGui::BeginCombo("Render Mode", currMode.c_str()))
		{
			if (ImGui::Selectable("Splats"))
				currMode = "Splats";
			if (ImGui::Selectable("Initial Points"))
				currMode = "Initial Points";
			if (ImGui::Selectable("Ellipsoids"))
				currMode = "Ellipsoids";
			ImGui::EndCombo();
		}
	}
	if (currMode == "Splats")

	// if (updateDebugPixelLocation && updateWithMouse)
	// {
	// 	auto viewWindowPos = sibr::getImGuiWindowPosition("Point view");
	// 	auto mousePos = ImGui::GetMousePos();
	// 	debugMode.debugPixel[0] = mousePos.x - viewWindowPos.x;
	// 	debugMode.debugPixel[1] = mousePos.y - viewWindowPos.y;
	// }

	{
		ImGui::SliderFloat("Scaling Modifier", &_scalingModifier, 0.001f, 1.0f);

		if (ImGui::CollapsingHeader("StopThePop", ImGuiTreeNodeFlags_DefaultOpen))
		{
			if (ImGui::BeginCombo("Sort Order", toString(splatting_settings.sort_settings.sort_order).c_str()))
			{
				if (ImGui::Selectable("VIEWSPACE_Z"))
					splatting_settings.sort_settings.sort_order = CudaRasterizer::GlobalSortOrder::VIEWSPACE_Z;
				if (ImGui::Selectable("DISTANCE"))
					splatting_settings.sort_settings.sort_order = CudaRasterizer::GlobalSortOrder::DISTANCE;
				if (ImGui::Selectable("PER_TILE_DEPTH_CENTER"))
					splatting_settings.sort_settings.sort_order = CudaRasterizer::GlobalSortOrder::PER_TILE_DEPTH_CENTER;
				if (ImGui::Selectable("PER_TILE_DEPTH_MAXPOS"))
					splatting_settings.sort_settings.sort_order = CudaRasterizer::GlobalSortOrder::PER_TILE_DEPTH_MAXPOS;
				ImGui::EndCombo();
			}

			if (ImGui::BeginCombo("Sort Mode", toString(splatting_settings.sort_settings.sort_mode).c_str()))
			{
				if (ImGui::Selectable("GLOBAL"))
					splatting_settings.sort_settings.sort_mode = CudaRasterizer::SortMode::GLOBAL;
				if (ImGui::Selectable("FULL SORT"))
					splatting_settings.sort_settings.sort_mode = CudaRasterizer::SortMode::PER_PIXEL_FULL;
				if (ImGui::Selectable("KBUFFER"))
					splatting_settings.sort_settings.sort_mode = CudaRasterizer::SortMode::PER_PIXEL_KBUFFER;
				if (ImGui::Selectable("HIERARCHICAL"))
					splatting_settings.sort_settings.sort_mode = CudaRasterizer::SortMode::HIERARCHICAL;
				ImGui::EndCombo();
			}

			if (splatting_settings.sort_settings.sort_mode == CudaRasterizer::SortMode::PER_PIXEL_KBUFFER)
			{
				if (ImGui::BeginCombo("Per-Pixel Queue Size", std::to_string(splatting_settings.sort_settings.queue_sizes.per_pixel).c_str()))
				{
					for (auto z : CudaRasterizer::per_pixel_queue_sizes){
						if (ImGui::Selectable(std::to_string(z).c_str()))
							splatting_settings.sort_settings.queue_sizes.per_pixel = z;
					}
					ImGui::EndCombo();
				}
			}

			if (splatting_settings.sort_settings.sort_mode == CudaRasterizer::SortMode::HIERARCHICAL)
			{
				if (ImGui::BeginCombo("Per-Pixel Queue Size", std::to_string(splatting_settings.sort_settings.queue_sizes.per_pixel).c_str()))
				{
					for (auto z : CudaRasterizer::per_pixel_queue_sizes_hier){
						if (ImGui::Selectable(std::to_string(z).c_str()))
							splatting_settings.sort_settings.queue_sizes.per_pixel = z;
					}
					ImGui::EndCombo();
				}

				if (ImGui::BeginCombo("2x2 Tile Queue Size", std::to_string(splatting_settings.sort_settings.queue_sizes.tile_2x2).c_str()))
				{
					for (auto z : CudaRasterizer::twobytwo_tile_queue_sizes){
						if (ImGui::Selectable(std::to_string(z).c_str()))
							splatting_settings.sort_settings.queue_sizes.tile_2x2 = z;
					}
					ImGui::EndCombo();
				}

				ImGui::Checkbox("Hier. 4x4 Tile Culling", &splatting_settings.culling_settings.hierarchical_4x4_culling);
			}


			ImGui::Checkbox("Foveated Rendering", &splatting_settings.foveated_rendering);
			ImGui::Checkbox("Rect Culling", &splatting_settings.culling_settings.rect_bounding);
			ImGui::Checkbox("Opacity Culling", &splatting_settings.culling_settings.tight_opacity_bounding);
			ImGui::Checkbox("Tile-based Culling", &splatting_settings.culling_settings.tile_based_culling);
			ImGui::Checkbox("Load Balancing", &splatting_settings.load_balancing);
			ImGui::Checkbox("Optimal Projection", &splatting_settings.optimal_projection);
			ImGui::Checkbox("Proper EWA Scaling", &splatting_settings.proper_ewa_scaling);
			ImGui::Checkbox("Blur", &blur);
		}


		if (ImGui::CollapsingHeader("Debug", ImGuiTreeNodeFlags_DefaultOpen))
		{
			if (ImGui::BeginCombo("Debug Visualization", toString(debugMode.type).data()))
			{
				if (ImGui::Selectable("Disabled"))
					debugMode.type = DebugVisualization::Disabled;
				if (ImGui::Selectable("Sort Error: Opacity Weighted"))
					debugMode.type = DebugVisualization::SortErrorOpacity;
				if (ImGui::Selectable("Sort Error: Distance Weighted"))
					debugMode.type = DebugVisualization::SortErrorDistance;
				if (ImGui::Selectable("Gaussian Count Per Tile"))
					debugMode.type = DebugVisualization::GaussianCountPerTile;
				if (ImGui::Selectable("Gaussian Count Per Pixel"))
					debugMode.type = DebugVisualization::GaussianCountPerPixel;
				if (ImGui::Selectable("Depth"))
					debugMode.type = DebugVisualization::Depth;
				if (ImGui::Selectable("Transmittance"))
					debugMode.type = DebugVisualization::Transmittance;
				ImGui::EndCombo();
			}

			if (debugMode.type != DebugVisualization::Disabled)
			{
				ImGui::Checkbox("Manual Normalization", &debugMode.debug_normalize);
				if (debugMode.debug_normalize)
				{
					ImGui::InputFloat2("Normalize Min/Max", debugMode.minMax);
				}
				ImGui::Checkbox("Input With Mouse", &updateWithMouse);
				ImGui::InputInt2("Mouse Debug Pos", debugMode.debugPixel);
			}

			ImGui::Checkbox("Timing", &debugMode.timing_enabled);
			if (debugMode.timing_enabled)
				ImGui::Text("%s", (char*) debugMode.timings_text.c_str());
			else
				debugMode.timings_text = "";
		}
		
	}
	ImGui::End();

	ImGui::Begin("Play");
		ImGui::SliderFloat("FPS", &s_fps, 1.0f, 120.0f); // Add FPS slider
		bool temp_play = _multi_view_play.load();

		// 2. Use the temporary 'bool' with ImGui.
		if (ImGui::Checkbox("multi view play", &temp_play)) {
			// 3. If the checkbox was clicked, update the atomic variable.
			_multi_view_play.store(temp_play);
			std::cout << "[GUI] Playback toggled to: " << (temp_play ? "ON" : "OFF") << std::endl;
		}

		int temp_frame_id = 0;
		{
			// Lock to safely read the current frame_id for the slider
			std::lock_guard<std::mutex> lock(mtx_frame_id);
			temp_frame_id = frame_id;
		}
		
		// Use the temporary variable for the slider
		if (ImGui::SliderInt("Playing Frame", &temp_frame_id, 0, (sequences_length - frame_st) / frame_step - 1)) {
			// If the slider was moved, lock and update the real frame_id
			{
				std::lock_guard<std::mutex> lock(mtx_frame_id);
				frame_id = temp_frame_id;
			}


			_multi_view_play = false; 
			need_download_q = std::queue<int>(); 
			need_ready_q = std::queue<int>(); 
			std::cout << "frame_id changed to " << frame_id << std::endl;
			
			// Reset tracker to force update
			last_loaded_frame_id = -1;
			
			int group_index = 0;
			if (num_frames > 0) { // Avoid division by zero if num_frames isn't set
				group_index = frame_id / num_frames;
			}
			if (group_index < group_frame_index.size()) {
				downloaded_frames = group_frame_index[group_index].second;
			} else {
				downloaded_frames = frame_id; // Fallback
			}
			ready_frames = frame_id; 
			
			for (int i = 0; i < sequences_length; i++) {
			
				if (downloaded_array[i] == 1) {
					std::cout << "Freeing downloaded frame " << i << std::endl;
					// free global png_vector to save memory
					for (int att_index = 0; att_index < num_att_index; att_index ++) {
						global_png_vector[att_index][i].release();;
					}
					downloaded_array[i] = 0;
				}

		
				if (ready_array[i] == 1) {
					// set ready to 0
					ready_array[i] = 0;
				}

			}
			
			group_index = frame_id / num_frames;
			int download_start_index = group_frame_index[group_index].first;
			int download_end_index = group_frame_index[group_index].second;

			need_download_q.push(group_index);
			cv_download.notify_one();

			for (int i = frame_id; i <= download_end_index; i+=frame_step)
			{	
				std::cout << "Preloading frame " << i << std::endl;
				need_ready_q.push(i);
				cv_ready.notify_one();
			}
			
			std::this_thread::sleep_for(std::chrono::milliseconds(3000));
			

			// request download and ready for frames around frame_id
			// request download 
			int target_group_index = frame_id / num_frames;
			for (int i = 1; i < download_cache_size + 1; i++) {
				if (target_group_index + i >= 0 && target_group_index + i < group_frame_index.size()) {
					int download_start_index = group_frame_index[target_group_index + i].first;
					if (downloaded_array[download_start_index] == 0) {
						std::cout << "[onGUI] Requesting group " << target_group_index + i << " to be downloaded." << std::endl;
						need_download_q.push(target_group_index + i);
					}
				}
			}
			cv_download.notify_one();
		}		
			
		float download_progress = sequences_length > 0 ? (float)downloaded_frames / (float)sequences_length : 0.0f;
		ImGui::ProgressBar(download_progress, ImVec2(-1.0f, 0.0f));
		ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
		ImGui::Text("Download Frame (%d / %d)", downloaded_frames, sequences_length);

		float ready_progress = sequences_length > 0 ? (float)ready_frames / (float)sequences_length : 0.0f;
		ImGui::ProgressBar(ready_progress, ImVec2(-1.0f, 0.0f));
		ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
		ImGui::Text("Ready Frame (%d / %d)", ready_frames, sequences_length);

	ImGui::End();

	if (ImGui::Combo("Remote Video list", &current_video_item, video_path.data(), video_path.size())) {
		std::cout << "Changing Video to: " << video_path[current_video_item] << std::endl;
		// empty all cuda memory
		for (int i = 0; i < sequences_length; i++) {
			if (ready_array[i] == 1) {
				// set ready to 0
				ready_array[i] = 0;
			}
		}
		frame_changed = true;
		_multi_view_play.store(false);
		cv_download.notify_all();
		cv_ready.notify_all();

		// Reset tracker on video switch
		last_loaded_frame_id = -1;

		std::cout << "Waiting for download thread to join..." << std::endl;
		if (download_thread_.joinable()) {
			download_thread_.join();
		}
		std::cout << "Waiting for ready thread to join..." << std::endl;
		if (ready_thread_.joinable()) {
			ready_thread_.join();
		}
		std::cout << "detect video changed, reset download thread and ready thread" << std::endl;
		
		// downloaded and ready frames need to be cleaned
		memset(downloaded_array, 0, sizeof(downloaded_array));
		memset(ready_array, 0, sizeof(ready_array));
		downloaded_frames = -1;
		ready_frames = -1;
		frame_id = 0;
		// clear group info
		group_frame_index.clear();
		group_name_index.clear();
		// clean global_png_vector
		for (int i = 0; i < num_att_index; i++) {
			global_png_vector[i].clear();
		}
		global_png_vector.clear();
		// clean the queue
		{
			std::lock_guard<std::mutex> lock(mtx_download);
			while (!need_download_q.empty()) {
				need_download_q.pop();
			}
		}
		{
			std::lock_guard<std::mutex> lock(mtx_ready);
			while (!need_ready_q.empty()) {
				need_ready_q.pop();
			}
		}
		// ready all cleand, as its directly converts to cuda mem
		
		// reset group info and folder and global_png_vector
		folder = video_path[current_video_item];
		_sh_degree = video_sh[current_video_item];
		num_att_index = (14 + (3 * (_sh_degree + 1) * (_sh_degree + 1))) + 3;
		std::string group_json_path = folder + "group_info.json";
		picojson::object group_obj = fetchJsonObj(group_json_path);
		std::string minmax_json_path = folder + "viewer_min_max.json";
		minmax_obj = fetchJsonObj(minmax_json_path);
		size_t num_groups = group_obj.size();
		for (int i = 0; i < num_groups; i++) {
			picojson::object innerObj = group_obj[std::to_string(i)].get<picojson::object>();
			picojson::array frame_index = innerObj["frame_index"].get<picojson::array>();
			group_frame_index.push_back(std::make_pair((int)frame_index[0].get<double>(), (int)frame_index[1].get<double>()));
			picojson::array name_index = innerObj["name_index"].get<picojson::array>();
			group_name_index.push_back(std::make_pair((int)name_index[0].get<double>(), (int)name_index[1].get<double>()));
		}
		global_png_vector.resize(num_att_index);
		sequences_length = group_frame_index[group_frame_index.size() - 1].second;
		std::cout << "New video length: " << sequences_length << std::endl;
		for (int i = 0; i < num_att_index; i++) {
			global_png_vector[i].resize(sequences_length + 1);
		}

		std::cout << "Reloading initial frames" << std::endl;

		// initial group index
		int group_index = 0;
		int download_start_index = group_frame_index[group_index].first;
		int download_end_index = group_frame_index[group_index].second;
		// test if the group is already downloaded
		std::vector<std::future<bool>> thread_futures;
		for (int att_index = 0; att_index < num_att_index; att_index ++) {
			std::string videopath = folder + "group" + std::to_string(group_index) + "/" + std::to_string(att_index) + ".mp4";
			thread_futures.push_back(std::async(std::launch::async, getAllFramesNew, videopath, download_start_index, std::ref(global_png_vector[att_index])));
		}
		for (auto& f : thread_futures) {
			bool success = f.get(); // This will wait for the thread to finish
			if (!success) {
				std::cerr << "Failed to process some videos" << std::endl;
			}
		}
		num_frames = download_end_index - download_start_index + 1;
		downloaded_frames = download_end_index;
		std::fill(downloaded_array + download_start_index, downloaded_array + download_end_index + 1, 1);
		// ready all frames in the group just need current to download end
		for (int i = download_start_index; i <= download_end_index; i+=frame_step) {
			loadVideo_func(i);
		}
		{
			// push all future undownloaded group index to queue
			std::lock_guard<std::mutex> lock(mtx_download);
			for (int i = 1; i < group_frame_index.size(); i++) {
				need_download_q.push(i);
			}
			cv_download.notify_one();
		}

		// restart the 2 threads
		frame_changed = false;
		download_thread_ = std::thread(&sibr::GaussianView::download_func, this);
		ready_thread_ = std::thread(&sibr::GaussianView::readyVideo_func, this);

		std::cout << "video changed to " << folder << " done" << std::endl;
	}

	ImGui::Checkbox("Fast culling", &_fastCulling);

	// visualize the data png
	static GLuint image_texture = 0;
	if (image_texture != 0) {
		glDeleteTextures(1, &image_texture);
		image_texture = 0;
	}
	glDeleteTextures(1, &image_texture);
	glGenTextures(1, &image_texture);
	glBindTexture(GL_TEXTURE_2D, image_texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	cv::Mat rgba_image(global_png_vector[1][frame_id].rows, global_png_vector[1][frame_id].cols, CV_8UC4);
	cv::cvtColor(global_png_vector[1][frame_id], rgba_image, cv::COLOR_GRAY2RGBA);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, rgba_image.cols, rgba_image.rows, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba_image.data);
	ImGui::Image((void*)(intptr_t)image_texture, ImVec2(rgba_image.cols, rgba_image.rows));

	ImGui::End();


	auto Memnow = std::chrono::high_resolution_clock::now();
	auto Memelapsed = std::chrono::duration_cast<std::chrono::milliseconds>(Memnow - MemlastUpdateTimestamp);
	if (Memelapsed > MemframeDuration) {
		auto [used, total] = gpuInfoManager.getMemoryUsage();
		Memused = used;
		Memtotal = total;
		MemlastUpdateTimestamp = Memnow;

		// net_speed_buffer
		auto total_bytes = getNetReceivedBytes();
		// divide memelapsed
		float speed = (total_bytes - last_total_bytes) * 8 / (Memelapsed.count() / 1000.f) / 1024 / 1024;
		last_total_bytes = total_bytes;
		net_speed_buffer.push_back(speed);
		if (net_speed_buffer.size() > 100) {
			net_speed_buffer.erase(net_speed_buffer.begin());
		}
	}
	ImGui::Begin("GPU Memory Usage");
	std::vector<float> memoryUsage = gpuInfoManager.getMemoryUsageBuffer();
	if (!memoryUsage.empty()) {
		ImGui::PlotLines("Memory Usage", memoryUsage.data(), memoryUsage.size(), 0, NULL, 0, 100, ImVec2(0, 80));
		ImGui::Text("Used: %.2f(GB) / Total: %.2f(GB)", float(Memused) / 1024.f / 1024.f / 1024.f, float(Memtotal) / 1024.f / 1024.f / 1024.f);
	}
	ImGui::End();

	// net
	ImGui::Begin("Network Usage");
	if (!net_speed_buffer.empty()) {
		// find the max speed
		float max_speed = *std::max_element(net_speed_buffer.begin(), net_speed_buffer.end());
		ImGui::PlotLines("Network Speed", net_speed_buffer.data(), net_speed_buffer.size(), 0, NULL, 0, max_speed, ImVec2(0, 80));
		ImGui::Text("Current Speed: %.2f(Mbps)", net_speed_buffer[net_speed_buffer.size() - 1]);
	}
	ImGui::End();

	if(!*_dontshow && !accepted && _interop_failed)
		ImGui::OpenPopup("Error Using Interop");

	if (!*_dontshow && !accepted && _interop_failed && ImGui::BeginPopupModal("Error Using Interop", NULL, ImGuiWindowFlags_AlwaysAutoResize)) {
		ImGui::SetItemDefaultFocus();
		ImGui::SetWindowFontScale(2.0f);
		ImGui::Text("This application tries to use CUDA/OpenGL interop.\n"\
			" It did NOT work for your current configuration.\n"\
			" For highest performance, OpenGL and CUDA must run on the same\n"\
			" GPU on an OS that supports interop.You can try to pass a\n"\
			" non-zero index via --device on a multi-GPU system, and/or try\n" \
			" attaching the monitors to the main CUDA card.\n"\
			" On a laptop with one integrated and one dedicated GPU, you can try\n"\
			" to set the preferred GPU via your operating system.\n\n"\
			" FALLING BACK TO SLOWER RENDERING WITH CPU ROUNDTRIP\n");

		ImGui::Separator();

		if (ImGui::Button("  OK  ")) {
			ImGui::CloseCurrentPopup();
			accepted = true;
		}
		ImGui::SameLine();
		ImGui::Checkbox("Don't show this message again", _dontshow);
		ImGui::EndPopup();
	}
}

sibr::GaussianView::~GaussianView()
{
	try {
		
		// Properly stop threads before destruction
		frame_changed = true;
		cv_download.notify_all();
		cv_ready.notify_all();
		
		// Try to join threads first, fallback to detach if needed
		try {
			if (download_thread_.joinable()) {
				// Give thread a reasonable time to finish
				auto future = std::async(std::launch::async, [&] {
					download_thread_.join();
				});
				if (future.wait_for(std::chrono::seconds(2)) == std::future_status::timeout) {
					SIBR_LOG << "Download thread did not finish in time, detaching..." << std::endl;
					download_thread_.detach();
				}
			}
		} catch (const std::exception& e) {
			SIBR_ERR << "Exception handling download thread in destructor: " << e.what() << std::endl;
			if (download_thread_.joinable()) {
				download_thread_.detach();
			}
		}
		
		try {
			if (ready_thread_.joinable()) {
				// Give thread a reasonable time to finish
				auto future = std::async(std::launch::async, [&] {
					ready_thread_.join();
				});
				if (future.wait_for(std::chrono::seconds(2)) == std::future_status::timeout) {
					SIBR_LOG << "Ready thread did not finish in time, detaching..." << std::endl;
					ready_thread_.detach();
				}
			}
		} catch (const std::exception& e) {
			SIBR_ERR << "Exception handling ready thread in destructor: " << e.what() << std::endl;
			if (ready_thread_.joinable()) {
				ready_thread_.detach();
			}
		}
		
		// CUDA cleanup
        for(int i = 0; i < GPU_RING_BUFFER_SLOTS; ++i)
        {
            CUDA_SAFE_CALL_ALWAYS(cudaStreamDestroy(data_streams[i]));
            CUDA_SAFE_CALL_ALWAYS(cudaEventDestroy(data_events[i]));
            CUDA_SAFE_CALL_ALWAYS(cudaFree(gpu_ring_buffer[i].pos_cuda));
            CUDA_SAFE_CALL_ALWAYS(cudaFree(gpu_ring_buffer[i].rot_cuda));
            CUDA_SAFE_CALL_ALWAYS(cudaFree(gpu_ring_buffer[i].scale_cuda));
            CUDA_SAFE_CALL_ALWAYS(cudaFree(gpu_ring_buffer[i].opacity_cuda));
            CUDA_SAFE_CALL_ALWAYS(cudaFree(gpu_ring_buffer[i].shs_cuda));
            CUDA_SAFE_CALL_ALWAYS(cudaFree(gpu_ring_buffer[i].rect_cuda));

			CUDA_SAFE_CALL_ALWAYS(cudaFree(gpu_ring_buffer[i].raw_attributes_cuda));
            CUDA_SAFE_CALL_ALWAYS(cudaFree(gpu_ring_buffer[i].minmax_values_cuda));
        }

        if (background_pos_cuda) {
            CUDA_SAFE_CALL_ALWAYS(cudaFree(background_pos_cuda));
            CUDA_SAFE_CALL_ALWAYS(cudaFree(background_rot_cuda));
            CUDA_SAFE_CALL_ALWAYS(cudaFree(background_scale_cuda));
            CUDA_SAFE_CALL_ALWAYS(cudaFree(background_opacity_cuda));
            CUDA_SAFE_CALL_ALWAYS(cudaFree(background_shs_cuda));
        }
        if (combined_pos_cuda) {
            CUDA_SAFE_CALL_ALWAYS(cudaFree(combined_pos_cuda));
            CUDA_SAFE_CALL_ALWAYS(cudaFree(combined_rot_cuda));
            CUDA_SAFE_CALL_ALWAYS(cudaFree(combined_scale_cuda));
            CUDA_SAFE_CALL_ALWAYS(cudaFree(combined_opacity_cuda));
            CUDA_SAFE_CALL_ALWAYS(cudaFree(combined_shs_cuda));
            CUDA_SAFE_CALL_ALWAYS(cudaFree(combined_rect_cuda));
        }
        CUDA_SAFE_CALL_ALWAYS(cudaStreamDestroy(combine_stream));
        CUDA_SAFE_CALL_ALWAYS(cudaEventDestroy(combine_event));

		cudaFree(view_cuda);
		cudaFree(proj_cuda);
		cudaFree(proj_inv_cuda);
		cudaFree(cam_pos_cuda);
		cudaFree(background_cuda);

		for (int i = 0; i < 2; i++)
			cudaFree(image_cuda_hier[i]);

		if (!_interop_failed)
		{
			cudaGraphicsUnregisterResource(imageBufferCuda);
		}
		else
		{
			cudaFree(fallbackBufferCuda);
		}
		glDeleteBuffers(1, &imageBuffer);

		if (geomPtr)
			cudaFree(geomPtr);
		if (binningPtr)
			cudaFree(binningPtr);
		if (imgPtr)
			cudaFree(imgPtr);

		delete _copyRenderer;
		
	} catch (const std::exception& e) {
		SIBR_ERR << "Exception in GaussianView destructor: " << e.what() << std::endl;
		// Continue with emergency cleanup
		try {
			if (download_thread_.joinable()) download_thread_.detach();
			if (ready_thread_.joinable()) ready_thread_.detach();
		} catch (...) {
			// Ignore further exceptions in destructor
		}
	}
}




