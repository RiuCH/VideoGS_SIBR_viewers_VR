// Software Name : SIBR_core
// SPDX-FileCopyrightText: Copyright (c) 2023 Orange
// SPDX-License-Identifier: Apache 2.0
//
// This software is distributed under the Apache 2.0 License;
// see the LICENSE file for more details.
//
// Author: Cédric CHEDALEUX <cedric.chedaleux@orange.com> et al.

#pragma once

namespace sibr
{
    enum class OpenXRInputType
    {
        LEFT_TRACKPAD_X,
        LEFT_TRACKPAD_Y,
        LEFT_TRACKPAD_CLICK,
        LEFT_TRIGGER,
        LEFT_THUMBSTICK_X,
        LEFT_THUMBSTICK_Y,
        LEFT_THUMBSTICK_CLICK,
        LEFT_AIM_POSE,
        RIGHT_TRACKPAD_X,
        RIGHT_TRACKPAD_Y,
        RIGHT_TRACKPAD_CLICK,
        RIGHT_TRIGGER,
        RIGHT_THUMBSTICK_X,
        RIGHT_THUMBSTICK_Y,
        RIGHT_THUMBSTICK_CLICK,
        RIGHT_AIM_POSE,
    };

}