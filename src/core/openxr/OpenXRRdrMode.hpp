// Software Name : SIBR_core
// SPDX-FileCopyrightText: Copyright (c) 2023 Orange
// SPDX-License-Identifier: Apache 2.0
//
// This software is distributed under the Apache 2.0 License;
// see the LICENSE file for more details.
//
// Author: Cédric CHEDALEUX <cedric.chedaleux@orange.com> et al.

#include <core/view/RenderingMode.hpp>
#include <core/graphics/Window.hpp>
#include <core/openxr/Config.hpp>
#include <core/openxr/OpenXRHMD.hpp>
#include <core/openxr/SwapchainImageRenderTarget.hpp>
#include <core/openxr/VRConfiguration.hpp>

#include <map>

namespace sibr
{

    /** OpenXRRdrMode renders a stereoscopic view to an Headset-Mouted display OpenXR device.
    *   It also renders both views to a SIBR view.
    *   \ingroup sibr_openxr
    */
    class SIBR_OPENXR_EXPORT OpenXRRdrMode : public IRenderingMode
    {
    public:
        /// Constructor.
        OpenXRRdrMode(sibr::Window &window, const std::string& configFile);
        ~OpenXRRdrMode();

        /** Perform rendering of a view.
         *\param view the view to render
         *\param eye the current camera
         *\param viewport the current viewport
         *\param optDest an optional destination RT
         */
        void render(ViewBase &view, const sibr::Camera &eye, const sibr::Viewport &viewport, IRenderTarget *optDest = nullptr);

        /** Get the current rendered image as a CPU image
         *\param current_img will contain the content of the RT */
        void destRT2img(sibr::ImageRGB &current_img)
        {
            return;
        }

        /** \return the left eye RT. */
        virtual const std::unique_ptr<RenderTargetRGB> &lRT() { return _leftRT; }
        /** \return the right eye RT. */
        virtual const std::unique_ptr<RenderTargetRGB> &rRT() { return _rightRT; }

        /** GUI for configuring OpenXR rendering */
        void onGui();

    private:
        std::unique_ptr<OpenXRHMD> m_openxrHmd;                  ///< OpenXR interface
        std::unique_ptr<VRConfiguration> m_vrConfig;             ///< Configuration for VR experience (initial camera pose and world transform)
        sibr::GLShader m_quadShader;                             ///< Shader for drawing left/right eye in desktop window
        std::map<int, SwapchainImageRenderTarget::Ptr> m_RTPool; ///< Pool for RenderTarget used to extract textures for each view
        int m_vrExperience = 0;                                  ///< 0: free world standing experience, 1: seated experience
        bool m_appFocused = false;                               ///< Application is visible and focused in the headset
        int m_downscaleResolution = 1;                           ///< Downscale rendering resolution to improve performance
        RenderTarget::UPtr _leftRT, _rightRT;                    ///< Only used to implement abstract method lRT andrRT!
        Camera m_headCameraInVrWorld;                            ///< Head pose camera in relative VR world coordinates
        Camera m_headCamera;                                     ///< Head pose camera in world coordinates
        bool m_rightTriggerPressed = false;                      ///< Right controller's trigger state
        bool m_leftTriggerPressed = false;                       ///< Left controller's trigger state
        Vector3f m_prevLeftHandPosition;                         ///< Store previous left hand position for drag scene transformation
        Quaternionf m_prevRightHandOrientation;                  ///< Store previous right hand position for drag scene rotation
        float m_controlSensitivity = 0.1f;                       ///< Control move/rotation sensitivity
        bool m_forceRenderVRPlaySpace = false;                   ///< Display or not the VR play space
        GLShader m_playSpaceShader;                              ///< Shader for drawing a red plane to represent the VR play space
        GLParameter m_playSpaceParamVP;                          ///< ViewProjection matrix uniform for play space shader
        GLParameter m_playSpaceBounds;                           ///< Play space width and height uniform for play space shader

        SwapchainImageRenderTarget::Ptr getRenderTarget(uint32_t texture, uint w, uint h);
        Vector3f vrToWorld(const Vector3f& pos) const;
        Quaternionf vrToWorld(const Quaternionf& quat) const;
        void updateHeadCamera(const Camera& camera);
        Camera computeEyeCam(const Camera& cam, OpenXRHMD::Eye eye) const;
        void renderVRPlaySpace(OpenXRHMD::Eye eye);

};

} /*namespace sibr*/