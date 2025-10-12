// Software Name : SIBR_core
// SPDX-FileCopyrightText: Copyright (c) 2023 Orange
// SPDX-License-Identifier: Apache 2.0
//
// This software is distributed under the Apache 2.0 License;
// see the LICENSE file for more details.
//
// Author: Cédric CHEDALEUX <cedric.chedaleux@orange.com> et al.

#include <core/openxr/interactionprofiles/HTCViveControllerProfile.hpp>

namespace sibr
{
    std::string HTCViveControllerProfile::getName() const {
        return "/interaction_profiles/htc/vive_controller";
    }

    // See specs: https://registry.khronos.org/OpenXR/specs/1.0/html/xrspec.html#_htc_vive_controller_profile
    std::map<OpenXRInputType, std::string> HTCViveControllerProfile::getBindings() const {
        return {
            {OpenXRInputType::LEFT_TRACKPAD_CLICK, "/user/hand/left/input/trackpad/click"},
            {OpenXRInputType::LEFT_TRACKPAD_X, "/user/hand/left/input/trackpad/x"},
            {OpenXRInputType::LEFT_TRACKPAD_Y, "/user/hand/left/input/trackpad/y"},
            {OpenXRInputType::LEFT_TRIGGER, "/user/hand/left/input/trigger/value"},
            {OpenXRInputType::LEFT_AIM_POSE, "/user/hand/left/input/aim/pose"},
            {OpenXRInputType::RIGHT_TRACKPAD_CLICK, "/user/hand/right/input/trackpad/click"},
            {OpenXRInputType::RIGHT_TRACKPAD_X, "/user/hand/right/input/trackpad/x"},
            {OpenXRInputType::RIGHT_TRACKPAD_Y, "/user/hand/right/input/trackpad/y"},
            {OpenXRInputType::RIGHT_TRIGGER, "/user/hand/right/input/trigger/value"},
            {OpenXRInputType::RIGHT_AIM_POSE, "/user/hand/right/input/aim/pose"}
        };
    }

}