// Software Name : SIBR_core
// SPDX-FileCopyrightText: Copyright (c) 2023 Orange
// SPDX-License-Identifier: Apache 2.0
//
// This software is distributed under the Apache 2.0 License;
// see the LICENSE file for more details.
//
// Author: Cédric CHEDALEUX <cedric.chedaleux@orange.com> et al.

#include <core/openxr/interactionprofiles/KhronosSimpleControllerProfile.hpp>

namespace sibr
{
    std::string KhronosSimpleControllerProfile::getName() const {
        return "/interaction_profiles/khr/simple_controller";
    }

    // See specs: https://registry.khronos.org/OpenXR/specs/1.0/html/xrspec.html#_khronos_simple_controller_profile
    std::map<OpenXRInputType, std::string> KhronosSimpleControllerProfile::getBindings() const {
        return {
            {OpenXRInputType::LEFT_AIM_POSE, "/user/hand/left/input/aim/pose"},
            {OpenXRInputType::RIGHT_AIM_POSE, "/user/hand/right/input/aim/pose"}
        };
    }

}