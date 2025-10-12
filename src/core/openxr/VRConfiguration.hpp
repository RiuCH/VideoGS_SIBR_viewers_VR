// Software Name : SIBR_core
// SPDX-FileCopyrightText: Copyright (c) 2023 Orange
// SPDX-License-Identifier: Apache 2.0
//
// This software is distributed under the Apache 2.0 License;
// see the LICENSE file for more details.
//
// Author: Cédric CHEDALEUX <cedric.chedaleux@orange.com> et al.

#pragma once

#include <string>

#include <core/openxr/Config.hpp>
#include <core/system/Transform3.hpp>
#include <core/system/Matrix.hpp>
#include <core/graphics/Camera.hpp>

namespace sibr
{

    /**
     *	Class that holds the configuration for the VR experience (initial camera pose and scene transform)
     *   \ingroup sibr_openxr
     */
    class VRConfiguration
    {
    public:
        explicit VRConfiguration(const std::string &filepath) : m_filePath(filepath) {}
        bool load();
        bool save() const;
        inline const std::string filePath() const {
             return m_filePath;
        }
        inline bool isSet() const
        {
            return m_isSet;
        }
        inline Camera &camera()
        {
            return m_camera;
        }
        inline const Camera &camera() const
        {
            return m_camera;
        }
        inline void setCamera(const Camera &camera)
        {
            m_camera = camera;
            m_isSet = true;
        }
        inline const Transform3f &sceneTransform() const
        {
            return m_sceneTransform;
        }
        inline Transform3f& sceneTransform()
        {
            return m_sceneTransform;
        }

    private:
        bool m_isSet = false;
        std::string m_filePath;

        // Origin of the camera for the VR experience
        sibr::Camera m_camera;

        // Transform to apply to the scene
        Transform3f m_sceneTransform;
    };
}