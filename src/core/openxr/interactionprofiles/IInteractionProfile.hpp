// Software Name : SIBR_core
// SPDX-FileCopyrightText: Copyright (c) 2023 Orange
// SPDX-License-Identifier: Apache 2.0
//
// This software is distributed under the Apache 2.0 License;
// see the LICENSE file for more details.
//
// Author: Cédric CHEDALEUX <cedric.chedaleux@orange.com> et al.

#pragma once

#include <map>
#include <string>
#include <core/openxr/OpenXRInputType.h>

namespace sibr
{
    class IInteractionProfile
    {
    public:
        virtual ~IInteractionProfile() {}
        virtual std::string getName() const = 0;
        virtual std::map<OpenXRInputType, std::string> getBindings() const = 0;
    };

}