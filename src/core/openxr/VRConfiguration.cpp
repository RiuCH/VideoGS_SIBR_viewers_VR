// Software Name : SIBR_core
// SPDX-FileCopyrightText: Copyright (c) 2023 Orange
// SPDX-License-Identifier: Apache 2.0
//
// This software is distributed under the Apache 2.0 License;
// see the LICENSE file for more details.
//
// Author: Cédric CHEDALEUX <cedric.chedaleux@orange.com> et al.

#include <core/openxr/VRConfiguration.hpp>
#include <core/system/Quaternion.hpp>

#include <picojson/picojson.hpp>
#include <fstream>

namespace sibr
{
    picojson::value vector3fToJsonValue(const sibr::Vector3f& vec) {
        picojson::array array {
            picojson::value(vec.x()),
            picojson::value(vec.y()),
            picojson::value(vec.z())
        };
        return picojson::value(array);
    }
    picojson::value quaternionToJsonValue(const sibr::Transform3f::Quaternion& q) {
        picojson::array array {
            picojson::value(q.x()),
            picojson::value(q.y()),
            picojson::value(q.z()),
            picojson::value(q.w())
        };
        return picojson::value(array);
    }

    sibr::Vector3f jsonArrayToVector3f(const picojson::array& array) {
        return Vector3f { (float) array[0].get<double>(), (float) array[1].get<double>(), (float) array[2].get<double>() };
    }

    sibr::Transform3f::Quaternion jsonArrayToQuaternion(const picojson::array& array) {
        return sibr::Transform3f::Quaternion {
            (float) array[3].get<double>(),
            (float) array[0].get<double>(),
            (float) array[1].get<double>(),
            (float) array[2].get<double>()
        };
    }


    bool VRConfiguration::load()
    {
        std::ifstream f(m_filePath);
        if (f.fail())
        {
            return false;
        }

        picojson::value v;
        picojson::set_last_error(std::string());
        std::string err = picojson::parse(v, f);
        if (!err.empty())
        {
            return false;
        }

        if (!v.contains("camera")) {
            SIBR_ERR << "Missing 'camera' attribute in VRConfig file" << std::endl;
            return false;
        }

        const auto& cameraV = v.get("camera");
        if (cameraV.contains("position")) {
            m_camera.position(jsonArrayToVector3f(cameraV.get("position").get<picojson::array>()));
        }
        if (cameraV.contains("rotation")) {
            m_camera.rotation(jsonArrayToQuaternion(cameraV.get("rotation").get<picojson::array>()));
        }
        if (cameraV.contains("znear")) {
            m_camera.znear((float) cameraV.get("znear").get<double>());
        }
        if (cameraV.contains("zfar")) {
            m_camera.zfar((float) cameraV.get("zfar").get<double>());
        }
        if (v.contains("sceneTransform")) {
            const auto& sceneTransformV = v.get("sceneTransform");
            m_sceneTransform.position(jsonArrayToVector3f(sceneTransformV.get("position").get<picojson::array>()));
            m_sceneTransform.rotation(jsonArrayToQuaternion(sceneTransformV.get("rotation").get<picojson::array>()));
        }
        f.close();

        m_isSet = true;
        return m_isSet;
    }

    bool VRConfiguration::save() const
    {
        std::ofstream f(m_filePath);
        if (f.fail())
        {
            return false;
        }

        picojson::object camera;
        camera.insert({"position", vector3fToJsonValue(m_camera.position())});
        camera.insert({"rotation", quaternionToJsonValue(m_camera.rotation())});
        camera.insert({"znear", picojson::value(m_camera.znear())});
        camera.insert({"zfar", picojson::value(m_camera.zfar())});

        picojson::object sceneTransform;
        sceneTransform.insert({"position", vector3fToJsonValue(m_sceneTransform.position())});
        sceneTransform.insert({"rotation", quaternionToJsonValue(m_sceneTransform.rotation())});

        picojson::object config;
        config.insert({"camera", picojson::value(camera)});
        config.insert({"sceneTransform", picojson::value(sceneTransform)});
        picojson::value v(config);

        f << v.serialize();
        f.close();
        return true;
    }

}