// Software Name : SIBR_core
// SPDX-FileCopyrightText: Copyright (c) 2023 Orange
// SPDX-License-Identifier: Apache 2.0
//
// This software is distributed under the Apache 2.0 License;
// see the LICENSE file for more details.
//
// Author: Cédric CHEDALEUX <cedric.chedaleux@orange.com> et al.

#pragma once

#include <core/openxr/interactionprofiles/IInteractionProfile.hpp>

namespace sibr
{
    class KhronosSimpleControllerProfile: public IInteractionProfile
    {
    public:
        KhronosSimpleControllerProfile() {};
        std::string getName() const override;
        std::map<OpenXRInputType, std::string> getBindings() const override;
    };

}