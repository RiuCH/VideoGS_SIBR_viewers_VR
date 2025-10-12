// Software Name : SIBR_core
// SPDX-FileCopyrightText: Copyright (c) 2023 Orange
// SPDX-License-Identifier: Apache 2.0
//
// This software is distributed under the Apache 2.0 License;
// see the LICENSE file for more details.
//
// Author: Cédric CHEDALEUX <cedric.chedaleux@orange.com> et al.

#include <core/openxr/interactionprofiles/OculusTouchControllerProfile.hpp>

namespace sibr
{
    std::string OculusTouchControllerProfile::getName() const {
        return "/interaction_profiles/oculus/touch_controller";
    }

    // See specs: https://registry.khronos.org/OpenXR/specs/1.0/html/xrspec.html#_oculus_touch_controller_profile
    std::map<OpenXRInputType, std::string> OculusTouchControllerProfile::getBindings() const {
        return {
            {OpenXRInputType::LEFT_THUMBSTICK_CLICK, "/user/hand/left/input/thumbstick/click"},
            {OpenXRInputType::LEFT_THUMBSTICK_X, "/user/hand/left/input/thumbstick/x"},
            {OpenXRInputType::LEFT_THUMBSTICK_Y, "/user/hand/left/input/thumbstick/y"},
            {OpenXRInputType::LEFT_TRIGGER, "/user/hand/left/input/trigger/value"},
            {OpenXRInputType::LEFT_AIM_POSE, "/user/hand/left/input/aim/pose"},
            {OpenXRInputType::RIGHT_THUMBSTICK_CLICK, "/user/hand/right/input/thumbstick/click"},
            {OpenXRInputType::RIGHT_THUMBSTICK_X, "/user/hand/right/input/thumbstick/x"},
            {OpenXRInputType::RIGHT_THUMBSTICK_Y, "/user/hand/right/input/thumbstick/y"},
            {OpenXRInputType::RIGHT_TRIGGER, "/user/hand/right/input/trigger/value"},
            {OpenXRInputType::RIGHT_AIM_POSE, "/user/hand/right/input/aim/pose"}
        };
    }

}