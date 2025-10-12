// Software Name : SIBR_core
// SPDX-FileCopyrightText: Copyright (c) 2023 Orange
// SPDX-License-Identifier: Apache 2.0
//
// This software is distributed under the Apache 2.0 License;
// see the LICENSE file for more details.
//
// Author: Cédric CHEDALEUX <cedric.chedaleux@orange.com> et al.

#include <core/graphics/GUI.hpp>
#include <core/assets/Resources.hpp>
#include <core/openxr/OpenXRRdrMode.hpp>
#include <core/openxr/SwapchainImageRenderTarget.hpp>

namespace sibr
{

#ifdef XR_USE_PLATFORM_XLIB
    XrGraphicsBindingOpenGLXlibKHR createXrGraphicsBindingOpenGLXlibKHR(Display *display, GLXDrawable drawable, GLXContext context)
    {
        return XrGraphicsBindingOpenGLXlibKHR{
            .type = XR_TYPE_GRAPHICS_BINDING_OPENGL_XLIB_KHR,
            .xDisplay = display,
            .glxDrawable = drawable,
            .glxContext = context};
    }
#endif

#ifdef XR_USE_PLATFORM_WIN32
    XrGraphicsBindingOpenGLWin32KHR createXrGraphicsBindingOpenGLWin32KHR(HDC hdc, HGLRC hglrc)
    {
        // Windows C++ compiler does not support C99 designated initializers
        return XrGraphicsBindingOpenGLWin32KHR{
            XR_TYPE_GRAPHICS_BINDING_OPENGL_WIN32_KHR, // .type
            NULL,                                      // .next
            hdc,                                       // .hDC
            hglrc                                      // .hGLRC
        };
    }
#endif

    OpenXRRdrMode::OpenXRRdrMode(sibr::Window &window, const std::string& configFile)
    {
        m_quadShader.init("Texture",
                          sibr::loadFile(sibr::Resources::Instance()->getResourceFilePathName("texture.vp")),
                          sibr::loadFile(sibr::Resources::Instance()->getResourceFilePathName("texture.fp")));

        // Shader to render a red quad at world ground (xz plane)
        std::string vertexShader =
            SIBR_SHADER(420,
                uniform mat4 viewProj;
                uniform vec2 bounds;
                out gl_PerVertex {
                    vec4 gl_Position;
                };

                // Generate 1-unit quad position on xz plane
                void main() {
                    vec2 pos = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2) - 1.0f;
	                gl_Position = viewProj * vec4(pos.x * bounds.x / 2.f, 0.f, pos.y * bounds.y / 2.f, 1.0);
                });
        std::string fragmentShader = SIBR_SHADER(420,
			out vec4 out_color;
		    void main(void) {
			    out_color = vec4(1.0, 0.0, 0.0, 0.8); // red semi-transparent quad
		    }
		);
		m_playSpaceShader.init("PlaySpace", vertexShader, fragmentShader);
		m_playSpaceParamVP.init(m_playSpaceShader,"viewProj");
		m_playSpaceBounds.init(m_playSpaceShader,"bounds");

        m_vrConfig = std::make_unique<VRConfiguration>(configFile);
        if (!m_vrConfig->load()) {
            SIBR_LOG << "No configuration file found for VR experience. Use default camera" << std::endl;
        }

        m_openxrHmd = std::make_unique<OpenXRHMD>("Gaussian splatting");
        if (!m_openxrHmd->init()) {
            SIBR_ERR << "Failed to connect to OpenXR" << std::endl;
        }

        bool sessionCreated = false;
#if defined(XR_USE_PLATFORM_XLIB)
        sessionCreated = m_openxrHmd->startSession(createXrGraphicsBindingOpenGLXlibKHR(glfwGetX11Display(), glXGetCurrentDrawable(), glfwGetGLXContext(window.GLFW())));
#elif defined(XR_USE_PLATFORM_WIN32)
        sessionCreated = m_openxrHmd->startSession(createXrGraphicsBindingOpenGLWin32KHR(wglGetCurrentDC(), wglGetCurrentContext()));
#endif
        if (!sessionCreated)
        {
            SIBR_ERR << "Failed to connect to OpenXR" << std::endl;
        }

        SIBR_LOG << "Disable VSync: use headset synchronization." << std::endl;
        window.setVsynced(false);

        m_openxrHmd->setIdleAppCallback([this]()
                                        { m_appFocused = false; });
        m_openxrHmd->setVisibleAppCallback([this]()
                                           { m_appFocused = false; });
        m_openxrHmd->setFocusedAppCallback([this]()
                                           { m_appFocused = true; });

        if (m_openxrHmd->input()) {
            // Move camera with left stick
            m_openxrHmd->input()->setStickMoveCallback(OpenXRInput::Hand::LEFT, [this](float x, float y) {
                float step = 0.1f;
                if (abs(x) > 0.5f) {
                    m_vrConfig->camera().translate((m_vrConfig->camera().rotation() * m_headCameraInVrWorld.right()) * x * step * m_controlSensitivity);
                }
                if (abs(y) > 0.5f) {
                    m_vrConfig->camera().translate((m_vrConfig->camera().rotation() * m_headCameraInVrWorld.dir()) * y * step * m_controlSensitivity);
                }
            });
            m_openxrHmd->input()->setStickMoveCallback(OpenXRInput::Hand::RIGHT, [this](float x, float y) {
                float step = 0.1f;
                // Rotate camera with right horizontal stick
                if (abs(x) > 0.5f) {
                    m_vrConfig->camera().rotate(Quaternionf(Eigen::AngleAxisf(- x * step * m_controlSensitivity, m_vrConfig->camera().up())));
                }
                // Elevate/lower camera with right vertical stick
                if (abs(y) > 0.5f) {
                    m_vrConfig->camera().translate((m_vrConfig->camera().rotation() * m_headCameraInVrWorld.up()) * y * step * m_controlSensitivity);
                }
            });
            // Move scene with left hand drag (position + trigger)
            m_openxrHmd->input()->setTriggerCallback(OpenXRInput::Hand::LEFT, [this](float val) {
                const Vector3f& handPose = vrToWorld(m_openxrHmd->input()->getHandPosePosition(OpenXRInput::Hand::LEFT));
                m_leftTriggerPressed = val > 0.5f;
                if (m_leftTriggerPressed)
                {
                    Vector3f t =  (handPose - m_prevLeftHandPosition);
                    m_vrConfig->sceneTransform().translate(t);
                }
                m_prevLeftHandPosition = handPose;

            });
            // Rotate scene with right hand drag (orientation + trigger)
            m_openxrHmd->input()->setTriggerCallback(OpenXRInput::Hand::RIGHT, [this](float val) {
                const Quaternionf& handRotation = vrToWorld(m_openxrHmd->input()->getHandPoseOrientation(OpenXRInput::Hand::RIGHT));
                m_rightTriggerPressed = val > 0.5f;
                if (m_rightTriggerPressed)
                {
                    Quaternionf diff = m_prevRightHandOrientation.slerp(0.1f, handRotation) * m_prevRightHandOrientation.inverse();
                    m_vrConfig->sceneTransform().rotate(diff.inverse());
                }
                m_prevRightHandOrientation = handRotation;
            });
        }
    }

    OpenXRRdrMode::~OpenXRRdrMode()
    {
        m_RTPool.clear();
        m_openxrHmd->closeSession();
        m_openxrHmd->terminate();
    }

    void OpenXRRdrMode::render(ViewBase &view, const sibr::Camera &camera, const sibr::Viewport &viewport, IRenderTarget *optDest)
    {
        // Render the UI with OpenXR infos
        onGui();

        if (!m_openxrHmd->isSessionRunning())
        {
            return;
        }

        // Get next pose prediction for rendering
        m_openxrHmd->pollEvents();
        if (!m_openxrHmd->waitNextFrame())
        {
            return;
        }

        // Headset pose has changed, let's update our VR head camera
        updateHeadCamera(camera);

        const int w = m_openxrHmd->getResolution().x();
        const int h = m_openxrHmd->getResolution().y();

        // Prepare the view to render at a specific resolution
        view.setResolution(sibr::Vector2i(w / m_downscaleResolution, h / m_downscaleResolution));

        // The callback is called for each single view (left view then right view) with the texture to render to
        m_openxrHmd->submitFrame([this, w, h, &view, optDest](int viewIndex, uint32_t texture)
                                 {
                                     OpenXRHMD::Eye eye = viewIndex == 0 ? OpenXRHMD::Eye::LEFT : OpenXRHMD::Eye::RIGHT;

                                     // Get the render target holding the swapchain image's texture from the pool
                                     auto rt = getRenderTarget(texture, w, h);
                                     if (!rt)
                                     {
                                         return;
                                     }

                                     // Compute eye with the parallax shift and asymetric fov
                                     Camera cam = computeEyeCam(m_headCamera, eye);

                                     // Perform the scene rendering for the given view into the RenderTarget's FBO
                                     rt->clear();
                                     rt->bind();
                                     glViewport(0, 0, w, h);
                                     view.onRenderIBR(*rt.get(), cam);
                                     rt->unbind();

                                     // Render the VR play space to help configuring the scene position and orientation
                                     if (m_forceRenderVRPlaySpace || m_leftTriggerPressed || m_rightTriggerPressed)
                                     {
                                        rt->bind();
                                        glViewport(0, 0, w, h);
                                        renderVRPlaySpace(eye);
                                        rt->unbind();
                                     }

                                     // Draw the left and right textures into the UI window
                                     if (optDest)
                                     {
                                         glViewport(eye == OpenXRHMD::Eye::LEFT ? 0 : optDest->w() / 2, 0, optDest->w() / 2, optDest->h());
                                         glScissor(eye == OpenXRHMD::Eye::LEFT ? 0 : optDest->w() / 2, 0, optDest->w() / 2, optDest->h());
                                         optDest->bind();
                                     }
                                     else
                                     {
                                         glViewport(eye == OpenXRHMD::Eye::LEFT ? 0 : w / 2, 0, w / 2, h);
                                         glScissor(eye == OpenXRHMD::Eye::LEFT ? 0 : w / 2, 0, w / 2, h);
                                     }
                                     glEnable(GL_SCISSOR_TEST);
                                     glDisable(GL_BLEND);
                                     glDisable(GL_DEPTH_TEST);
                                     glClearColor(0.f, 0.f, 0.f, 1.f);
                                     glClear(GL_COLOR_BUFFER_BIT);
                                     m_quadShader.begin();
                                     glActiveTexture(GL_TEXTURE0);
                                     glBindTexture(GL_TEXTURE_2D, texture);
                                     RenderUtility::renderScreenQuad();
                                     glBindTexture(GL_TEXTURE_2D, 0);
                                     m_quadShader.end();
                                     glDisable(GL_SCISSOR_TEST);
                                     if (optDest)
                                     {
                                         optDest->unbind();
                                     }
                                 });
    }

    void OpenXRRdrMode::onGui()
    {
        const std::string guiName = "OpenXR";
        ImGui::Begin(guiName.c_str());
        std::string status = "KO";
        if (m_openxrHmd->isSessionRunning())
        {
            status = m_appFocused ? "FOCUSED" : "IDLE";
        }
        ImGui::Text("Session status: %s", status.c_str());
        ImGui::Text("Runtime: %s (%s)", m_openxrHmd->getRuntimeName().c_str(), m_openxrHmd->getRuntimeVersion().c_str());
        ImGui::Text("Reference space type: %s", m_openxrHmd->getReferenceSpaceType());
        ImGui::RadioButton("Free world standing", &m_vrExperience, 0);
        ImGui::SameLine();
        ImGui::RadioButton("Seated", &m_vrExperience, 1);
        if (m_openxrHmd->isSessionRunning())
        {
            const auto report = m_openxrHmd->getRefreshReport();
            ImGui::Text("Framerate: %.2f FPS (expected: %.2f FPS)", report.measuredFramerate, report.expectedFramerate);
            const auto w = m_openxrHmd->getResolution().x();
            const auto h = m_openxrHmd->getResolution().y();
            ImGui::Text("Headset resolution (per eye): %ix%i", w, h);
            ImGui::Text("Rendering resolution (per eye): %ix%i", w / m_downscaleResolution, h / m_downscaleResolution);
			ImGui::SliderInt("Down scale factor", &m_downscaleResolution, 1, 8);
            ImGui::Text("IPD: %.1fcm", m_openxrHmd->getInterPupillaryDistance() * 100.f);
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenOverlapped))
            {
                ImGui::SetTooltip("Inter-pupillary distance");
            }
            ImGui::Checkbox("Show VR play space", &m_forceRenderVRPlaySpace);
            if (ImGui::Button("Save VR configuration"))
            {
                if (m_vrConfig->save())
                {
                    SIBR_LOG << "VR configuration saved to '" << m_vrConfig->filePath() << "'" << std::endl;
                }
            }
            if (ImGui::CollapsingHeader("Controls"))
            {
			    ImGui::SliderFloat("Sensitivity", &m_controlSensitivity, 0.1f, 1.f);
                ImGui::Text("Move camera: Left stick or");
                ImGui::SameLine();
                if (ImGui::Button("Left"))
                {
                    m_vrConfig->camera().translate((m_vrConfig->camera().rotation() * m_headCameraInVrWorld.right() * 1.0f) * m_controlSensitivity);
                }
                ImGui::SameLine();
                if (ImGui::Button("Right"))
                {
                    m_vrConfig->camera().translate((m_vrConfig->camera().rotation() * m_headCameraInVrWorld.right()) * m_controlSensitivity);
                }
                ImGui::SameLine();
                if (ImGui::Button("Forward"))
                {
                    m_vrConfig->camera().translate((m_vrConfig->camera().rotation() * m_headCameraInVrWorld.dir()) * m_controlSensitivity);
                }
                ImGui::SameLine();
                if (ImGui::Button("Backward"))
                {
                    m_vrConfig->camera().translate((m_vrConfig->camera().rotation() * m_headCameraInVrWorld.dir() * 1.0f) * m_controlSensitivity);
                }
                ImGui::Text("Elevate/lower camera: Right vertical stick or");
                ImGui::SameLine();
                if (ImGui::Button("Up"))
                {
                    m_vrConfig->camera().translate(m_vrConfig->camera().up() * m_controlSensitivity);
                }
                ImGui::SameLine();
                if (ImGui::Button("Down"))
                {
                    m_vrConfig->camera().translate(-m_vrConfig->camera().up() * m_controlSensitivity);
                }
                ImGui::Text("Rotate camera: Right horizontal stick or");
                ImGui::SameLine();
                if (ImGui::Button("Rotate CW"))
                {
                    m_vrConfig->camera().rotate(Quaternionf(Eigen::AngleAxisf(-m_controlSensitivity, m_vrConfig->camera().up())));
                }
                ImGui::SameLine();
                if (ImGui::Button("Rotate ACW"))
                {
                    m_vrConfig->camera().rotate(Quaternionf(Eigen::AngleAxisf(m_controlSensitivity, m_vrConfig->camera().up())));
                }
                ImGui::Text("Move scene: drag with left controller");
                ImGui::Text("Rotate scene: drag with right controller");
            }
            if (ImGui::CollapsingHeader("Debug"))
            {
                ImGui::Text("Left eye:");
                const auto leftPos = this->m_openxrHmd->getPosePosition(OpenXRHMD::Eye::LEFT);
                const auto leftRot = this->m_openxrHmd->getPoseOrientation(OpenXRHMD::Eye::LEFT, OpenXRHMD::AngleUnit::DEGREE);
                const auto rightPos = this->m_openxrHmd->getPosePosition(OpenXRHMD::Eye::RIGHT);
                const auto rightRot = this->m_openxrHmd->getPoseOrientation(OpenXRHMD::Eye::RIGHT, OpenXRHMD::AngleUnit::DEGREE);
                auto fov = m_openxrHmd->getFieldOfView(OpenXRHMD::Eye::LEFT, OpenXRHMD::AngleUnit::DEGREE);
                ImGui::Text("\tFOV: %.2f°, %.2f°, %.2f°, %.2f°", fov.x(), fov.y(), fov.z(), fov.w());
                ImGui::Text("\tPosition : %.2f, %.2f, %.2f", leftPos.x(), leftPos.y(), leftPos.z());
                ImGui::Text("\tOrientation : %.2f, %.2f, %.2f", leftRot.x(), leftRot.y(), leftRot.z());
                ImGui::Text("Right eye:");
                fov = m_openxrHmd->getFieldOfView(OpenXRHMD::Eye::RIGHT, OpenXRHMD::AngleUnit::DEGREE);
                ImGui::Text("\tFOV: %.2f°, %.2f°, %.2f°, %.2f°", fov.x(), fov.y(), fov.z(), fov.w());
                ImGui::Text("\tPosition : %.2f, %.2f, %.2f", rightPos.x(), rightPos.y(), rightPos.z());
                ImGui::Text("\tOrientation : %.2f, %.2f, %.2f", rightRot.x(), rightRot.y(), rightRot.z());
            }
        }
    }

    SwapchainImageRenderTarget::Ptr OpenXRRdrMode::getRenderTarget(uint32_t texture, uint w, uint h)
    {
        auto i = m_RTPool.find(texture);
        if (i != m_RTPool.end())
        {
            return i->second;
        }
        else
        {
            SwapchainImageRenderTarget::Ptr newRt = std::make_shared<SwapchainImageRenderTarget>(texture, w, h);
            auto pair = m_RTPool.insert(std::make_pair<int, SwapchainImageRenderTarget::Ptr>(texture, std::move(newRt)));
            if (pair.second)
            {
                return (*pair.first).second;
            }
        }
        return SwapchainImageRenderTarget::Ptr();
    }

    Camera createCamera(float znear, float zfar, const Vector3f& position, const Quaternionf& rotation) {
        Camera cam;
        cam.znear(znear);
        cam.zfar(zfar);
        cam.position(position);
        cam.rotation(rotation);
        return cam;
    }

    Vector3f OpenXRRdrMode::vrToWorld(const Vector3f& pos) const {
        return m_vrConfig->camera().rotation() * pos + m_vrConfig->camera().position();
    }

    Quaternionf OpenXRRdrMode::vrToWorld(const Quaternionf& quat) const {
        return m_vrConfig->camera().rotation() * quat;
    }
    
    void OpenXRRdrMode::updateHeadCamera(const Camera& camera)
    {
        // If not set, initialize the VR configuration from the current camera
        if (!m_vrConfig->isSet())
        {
            Camera origin = camera;
             // ignore current camera orientation but flip Y and Z to respect SIBR camera coordinate convention (180° rotation about x)
            origin.rotation(Quaternionf { 0.f, 1.0f, 0.f, 0.f});
            m_vrConfig->setCamera(origin);
            m_vrConfig->sceneTransform().set(Vector3f(), Quaternionf::Identity());
        }

        auto znear = m_vrConfig->camera().znear();
        auto zfar = m_vrConfig->camera().zfar();

        // Cam (in VR world coordinates) used for VR ground rendering
        m_headCameraInVrWorld = createCamera(znear, zfar, m_openxrHmd->getHeadPosePosition(), m_openxrHmd->getHeadPoseQuaternion());

        // Transform relative VR pose to world coordinates in respect to VR config inital camera
        Vector3f headPosition;
        if (m_vrExperience == 0) { // free world => full VR experience
            headPosition = vrToWorld(m_headCameraInVrWorld.position());
            // Make head y position only depends on headpose height to not cumulate with the vr config camera's height
            headPosition.y() -= m_vrConfig->camera().position().y();
        } else { // seated => ignore headset position (video 360 experience)
            headPosition = m_vrConfig->camera().position();
        }
        Quaternionf headRotation = m_vrConfig->camera().rotation() * m_headCameraInVrWorld.rotation();

        // Apply scene transform
        Quaternionf q = m_vrConfig->sceneTransform().rotation().inverse();
        Vector3f t = -1.0f * (q * m_vrConfig->sceneTransform().position());
        // Cam with scene transform in world coordinates (used for rendering)
        m_headCamera = createCamera(znear, zfar, q * headPosition + t, q * headRotation);
    }


    Camera OpenXRRdrMode::computeEyeCam(const Camera& cam, OpenXRHMD::Eye eye) const {
        Camera eyeCam = cam;
        Vector3f parallaxShift = m_openxrHmd->getInterPupillaryDistance() / 2.f * eyeCam.right();
        if (eye == OpenXRHMD::Eye::LEFT)
        {
            parallaxShift *= -1.f;
        }
        eyeCam.translate(parallaxShift);
        const auto& fov = m_openxrHmd->getFieldOfView(eye);
        eyeCam.fovy(fov.w() - fov.z());
        eyeCam.aspect((fov.y() - fov.x()) / (fov.w() - fov.z()));
        // Note: setStereoCam() used in SteroAnaglyph cannot be reused here,
        // because headset eye views have asymetric fov
        // We therefore use the perspective() method with principal point positioning instead
        eyeCam.principalPoint(Eigen::Vector2f(1.f, 1.f) - m_openxrHmd->getScreenCenter(eye));
        eyeCam.perspective(eyeCam.fovy(), eyeCam.aspect(), eyeCam.znear(), eyeCam.zfar());
        return eyeCam;
    }

    void OpenXRRdrMode::renderVRPlaySpace(OpenXRHMD::Eye eye) {
        Camera vrCam = computeEyeCam(m_headCameraInVrWorld, eye);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        m_playSpaceShader.begin();
        m_playSpaceParamVP.set(vrCam.viewproj());
        m_playSpaceBounds.set(Vector2f {m_openxrHmd->getPlaySpaceBounds().x(), m_openxrHmd->getPlaySpaceBounds().y()});
        RenderUtility::useDefaultVAO();
        const unsigned char indices[] = {0, 1, 2, 3};
        glDrawElements(GL_TRIANGLE_STRIP, 4, GL_UNSIGNED_BYTE, indices);
        CHECK_GL_ERROR;
        m_playSpaceShader.end();
        glDisable(GL_BLEND);
    }

} /*namespace sibr*/