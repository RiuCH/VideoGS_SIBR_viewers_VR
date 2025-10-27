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
#pragma once

# include "Config.hpp"
# include <core/renderer/RenderMaskHolder.hpp>
# include <core/scene/BasicIBRScene.hpp>
# include <core/system/SimpleTimer.hpp>
# include <core/system/Config.hpp>
# include <core/graphics/Mesh.hpp>
# include <core/view/ViewBase.hpp>
# include <core/renderer/CopyRenderer.hpp>
# include <core/renderer/PointBasedRenderer.hpp>
# include <memory>
# include <core/graphics/Texture.hpp>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <functional>
# include "GaussianSurfaceRenderer.hpp"
#include "GSVideoDecoder.hpp"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <vector>
#include <thread>
# include <queue>
# include <mutex>
# include <condition_variable>
#include <picojson/picojson.hpp>
#include "GPUMemoryInfo.hpp"
#include <atomic>
#include <future>
#ifdef __linux__
#include <sys/sysinfo.h>
#endif
#include <fstream> // Added for PLY loading

#include "kernels.hpp"

#define SEQUENCE_LENGTH 1000

#define MAX_GAUSSIANS_PER_FRAME 1500000 

#define GPU_RING_BUFFER_SLOTS 15

namespace CudaRasterizer
{
	class Rasterizer;
}

namespace sibr { 

	class BufferCopyRenderer;
	class BufferCopyRenderer2;

	struct GpuFrameSlot {
        float* pos_cuda = nullptr;
        float* rot_cuda = nullptr;
        float* scale_cuda = nullptr;
        float* opacity_cuda = nullptr;
        float* shs_cuda = nullptr;
        int* rect_cuda = nullptr;

		uint8_t* raw_attributes_cuda = nullptr;
        float* minmax_values_cuda = nullptr;
    };

	/**
	 * \class RemotePointView
	 * \brief Wrap a ULR renderer with additional parameters and information.
	 */
	class SIBR_EXP_ULR_EXPORT GaussianView : public sibr::ViewBase
	{
		SIBR_CLASS_PTR(GaussianView);

	public:

		/**
		 * Constructor
		 * \param ibrScene The scene to use for rendering.
		 * \param render_w rendering width
		 * \param render_h rendering height
		 * \param file Path to the static background PLY file (can be nullptr or empty)
		 */
		GaussianView(const sibr::BasicIBRScene::Ptr& ibrScene, uint render_w, uint render_h, const char* file, bool* message_read, bool white_bg = false, bool useInterop = true, int device = 0);

		/** Replace the current scene.
		 *\param newScene the new scene to render */
		void setScene(const sibr::BasicIBRScene::Ptr & newScene);

		/**
		 * Perform rendering. Called by the view manager or rendering mode.
		 * \param dst The destination rendertarget.
		 * \param eye The novel viewpoint.
		 */
		void onRenderIBR(sibr::IRenderTarget& dst, const sibr::Camera& eye) override;

		/**
		 * Update inputs (do nothing).
		 * \param input The inputs state.
		 */
		void onUpdate(Input& input) override;

		/**
		 * Update the GUI.
		 */
		void onGUI() override;

		/** \return a reference to the scene */
		const std::shared_ptr<sibr::BasicIBRScene> & getScene() const { return _scene; }

		virtual ~GaussianView() override;

		void setResolution(const Vector2i& size) override;

		bool* _dontshow;

		void download_func();
		void readyVideo_func();
		void loadVideo_func(int frame_index);

	protected:

        /** Load static background Gaussians from a PLY file. */
        void loadBackground(const std::string& ply_path);
		std:: string _background_ply_path = "/home/riu/Desktop/VideoGSProject/datasets/atc1_bg.ply";
		float background_scale = 100.0f;

		std::string currMode = "Splats";
		int _sh_degree = 1;
		int frame_st = 0;
		int frame_step = 1;
		int frame_id = 0;
		bool _fastCulling = true;
		int _device = 0;
		
		// Thread safety and resource management
		std::atomic<bool> _threads_initialized{false};
		std::mutex _thread_management_mutex;
		bool checkSystemResources();
		int num_att_index = 29;
		int sequences_length = 0;

		int ready_cache_size = 15;
		int download_cache_size = 10;
		std::mutex mtx_frame_id;

		int count; // Total count (background + dynamic) for rendering
		// Pointers to the *combined* buffers for rendering
		float* pos_cuda;
		float* rot_cuda;
		float* scale_cuda;
		float* opacity_cuda;
		float* shs_cuda;
		int* rect_cuda;
		int P_array[SEQUENCE_LENGTH]; // Count for *dynamic* frames

		GpuFrameSlot gpu_ring_buffer[GPU_RING_BUFFER_SLOTS];
        cudaStream_t data_streams[GPU_RING_BUFFER_SLOTS];
        cudaEvent_t data_events[GPU_RING_BUFFER_SLOTS];

        // --- Background Data Buffers (Static) ---
        int background_count = 0;
        float* background_pos_cuda = nullptr;
        float* background_rot_cuda = nullptr;
        float* background_scale_cuda = nullptr;
        float* background_opacity_cuda = nullptr;
        float* background_shs_cuda = nullptr;

        // --- Combined Data Buffers (for rendering) ---
        int combined_buffer_allocated_count = 0; // Max size allocated
        float* combined_pos_cuda = nullptr;
        float* combined_rot_cuda = nullptr;
        float* combined_scale_cuda = nullptr;
        float* combined_opacity_cuda = nullptr;
        float* combined_shs_cuda = nullptr;
        int* combined_rect_cuda = nullptr; // Needed if dynamic part uses fast culling

        cudaStream_t combine_stream = 0; // Stream for combining data
        cudaEvent_t combine_event = 0;   // Event to signal combination complete
	
		// std::vector<cv::Mat> png_vector;
		std::vector<std::vector<cv::Mat>> global_png_vector;
		// init decoder
		GSVideoDecoder decoder;

		// init memory info
		GPUInfoManager gpuInfoManager;
		float Memused, Memtotal, Memusage;

		// init network speed info
		unsigned long long last_total_bytes = 0;
		std::vector<float> net_speed_buffer;
		// std::string folder = "http://10.15.89.67:10000/video_new_new/";
		// std::string folder = "http://10.15.89.67:10000/video_new2_qp/video_qp20/";
		// std::string folder = "http://10.15.89.67:10000/coser18_qp15_new/";
		// std::string folder = "http://10.15.89.67:10000/1015hanfu_qp0/";
		std::string folder;

		int current_video_item = 0;
	std::vector<const char*> video_path = {
			// "http://10.15.89.67:10000/wyw_hifivv/",
			// "http://10.15.89.67:10000/hanfu3_qp15/",
			// "http://10.15.89.67:10000/0923dancer3/",
			// "http://10.15.89.67:10000/jywq_demo/",
			// "http://10.15.89.67:10000/ykx_boxing_long_qp15/",
			// "http://10.15.89.67:10000/jywq_qp15/",
			// "http://10.15.89.67:10000/coser18_0503_qp15/",
			// "http://10.15.89.67:10000/png_all_0/", 
			// "http://10.15.89.67:10000/coser18_qp15_new/", 
			// "http://10.15.89.67:10000/coser18_0503_qp0/", 
			// "http://10.15.89.67:10000/1015hanfu_qp0/", 
			// "http://10.15.89.67:10000/coser18_qp0_new/"
			// "http://127.0.0.1/bicycle_static/",
			// "http://127.0.0.1/atc_1_q0/",
			// "http://127.0.0.1/atc_1_q25/",
			// "http://127.0.0.1/atc_1_q25_nobg/",
			"http://127.0.0.1/atc_1_q0_nobg_full/",
			"http://127.0.0.1/atc_1_q0_nobg/",
			// "http://127.0.0.1/png_all_0/",
			// "http://127.0.0.1/png_all_0/",
			"http://127.0.0.1/png_all_25/",
			"http://127.0.0.1/png_all_50/",
			"http://127.0.0.1/ykx_boxing_long_qp15_380/"

		};
		std::vector<int> video_sh = {
			// 3,
			0,
			0,
			0,
			0,
			0,
			0,
			0,
			0,
			1,
			1,
			3,
			1,
			3,
			3
		};

		std::chrono::milliseconds frameDuration; // 33ms per frame -> 30fps for dynamic play
		std::chrono::high_resolution_clock::time_point lastUpdateTimestamp;

		std::chrono::milliseconds MemframeDuration; // read mem every 1s
		std::chrono::high_resolution_clock::time_point MemlastUpdateTimestamp;

		picojson::object minmax_obj;

		// multi thread helper
		std::vector<std::pair<int, int>> group_frame_index;
		std::vector<std::pair<int, int>> group_name_index;
		int downloaded_frames = -1;
		int downloaded_array[SEQUENCE_LENGTH];
		int ready_frames = -1;
		int ready_array[SEQUENCE_LENGTH];
		int num_frames = 1;

		bool frame_changed = false;

		// ready queue note that the index is frame index
		std::queue<int> need_ready_q;
		std::mutex mtx_ready;
		std::condition_variable cv_ready;

		// note the index is group index
		std::queue<int> need_download_q;
		std::mutex mtx_download;
		std::condition_variable cv_download;
		std::thread download_thread_;
		std::thread ready_thread_;

		GLuint imageBuffer = 0;
		cudaGraphicsResource_t imageBufferCuda;

		size_t allocdGeom = 0, allocdBinning = 0, allocdImg = 0;
		void* geomPtr = nullptr, * binningPtr = nullptr, * imgPtr = nullptr;
		std::function<char* (size_t N)> geomBufferFunc, binningBufferFunc, imgBufferFunc;

		float* view_cuda;
		float* proj_cuda;
		float* cam_pos_cuda;
		float* background_cuda;

		float _scalingModifier = 1.0f;
		GaussianData* gData;
		GaussianData* gData_array[500];
		// bool _multi_view_play = true;
		std::atomic<bool> _multi_view_play{false};\
		std::atomic<bool> _slider_seek_active{false};
		bool _use_interop = true;
		bool _interop_failed = false;
		std::vector<char> fallback_bytes;
		float* fallbackBufferCuda = nullptr;
		bool accepted = false;


		std::shared_ptr<sibr::BasicIBRScene> _scene; ///< The current scene.
		PointBasedRenderer::Ptr _pointbasedrenderer;
		BufferCopyRenderer* _copyRenderer;
		GaussianSurfaceRenderer* _gaussianRenderer;

		void createImageBuffer();
		void destroyImageBuffer();

	};

} /*namespace sibr*/
