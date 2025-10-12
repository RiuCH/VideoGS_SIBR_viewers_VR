// Software Name : SIBR_core
// SPDX-FileCopyrightText: Copyright (c) 2023 Orange
// SPDX-License-Identifier: Apache 2.0
//
// This software is distributed under the Apache 2.0 License;
// see the LICENSE file for more details.
//
// Author: Cédric CHEDALEUX <cedric.chedaleux@orange.com> et al.

#include <core/openxr/OpenXRHelper.hpp>
#include <core/openxr/OpenXRInput.hpp>
#include <core/openxr/interactionprofiles/KhronosSimpleControllerProfile.hpp>
#include <core/openxr/interactionprofiles/HTCViveControllerProfile.hpp>
#include <core/openxr/interactionprofiles/OculusTouchControllerProfile.hpp>
#include <core/openxr/interactionprofiles/ValveIndexControllerProfile.hpp>

#include <map>

namespace sibr
{
    OpenXRInput::OpenXRInput(XrInstance &instance, XrSession &session, XrSpace& refSpace) : m_instance(instance),
                                                                         m_session(session),
                                                                         m_refSpace(refSpace)
    {
        if (!createActions())
        {
            SIBR_ERR << "Failed to create OpenXR actions" << std::endl;
        }

        registerInteractionProfile(KhronosSimpleControllerProfile());
        registerInteractionProfile(HTCViveControllerProfile());
        registerInteractionProfile(OculusTouchControllerProfile());
        registerInteractionProfile(ValveIndexControllerProfile());

        if (!attachActionSet())
        {
            SIBR_ERR << "Failed to attach OpenXR action set" << std::endl;
        }
    }

    OpenXRInput::~OpenXRInput()
    {
        XrResult result;
        for (auto action : m_xrActions)
        {
            result = xrDestroyAction(action.second);
            xrCheck(m_instance, result, "Failed to destroy action!");
        }
        m_xrActions.clear();

        result = xrDestroyActionSet(m_actionSet);
        xrCheck(m_instance, result, "Failed to destroy action set!");
    }

    bool OpenXRInput::sync(XrTime time)
    {
        XrActiveActionSet activeActionSet;
        activeActionSet.actionSet = m_actionSet;
        activeActionSet.subactionPath = XR_NULL_PATH;

        XrActionsSyncInfo actionsSyncInfo;
        actionsSyncInfo.type = XR_TYPE_ACTIONS_SYNC_INFO;
        actionsSyncInfo.next = NULL;
        actionsSyncInfo.countActiveActionSets = 1;
        actionsSyncInfo.activeActionSets = &activeActionSet;
        XrResult result = xrSyncActions(m_session, &actionsSyncInfo);
        if (!xrCheck(m_instance, result, "failed to sync actions!"))
        {
            return false;
        }

        // Extract all input events and dispatch callbacks
        handleTrackpadEvent(Hand::LEFT);
        handleTrackpadEvent(Hand::RIGHT);
        handleThumbstickEvent(Hand::LEFT);
        handleThumbstickEvent(Hand::RIGHT);
        handleTriggerEvent(Hand::LEFT);
        handleTriggerEvent(Hand::RIGHT);

        locateSpaces(time);

        return true;
    }

    void OpenXRInput::setStickMoveCallback(Hand hand, const std::function<void(float, float)> &callback)
    {
        m_stickMoveCallback[static_cast<int>(hand)] = callback;
    }

    void OpenXRInput::setStickClickCallback(Hand hand, const std::function<void()> &callback)
    {
        m_stickClickCallback[static_cast<int>(hand)] = callback;
    }

    void OpenXRInput::setTriggerCallback(Hand hand, const std::function<void(float value)> &callback)
    {
        m_triggerCallback[static_cast<int>(hand)] = callback;
    }


    Vector3f OpenXRInput::getHandPosePosition(Hand hand) const {
        auto position = m_handPose[static_cast<int>(hand)].position;
        return Vector3f {position.x, position.y, position.z};
    }

    Quaternionf OpenXRInput::getHandPoseOrientation(Hand hand) const {
        auto orientation = m_handPose[static_cast<int>(hand)].orientation;
        return Quaternionf {orientation.w, orientation.x, orientation.y, orientation.z};
    }

    bool OpenXRInput::createActions()
    {
        XrResult result;

        // Create action set
        XrActionSetCreateInfo actionSetInfo;
        actionSetInfo.type = XR_TYPE_ACTION_SET_CREATE_INFO;
        actionSetInfo.next = NULL;
        actionSetInfo.priority = 0;
        stringCopy(actionSetInfo.actionSetName, "gameplay_default", XR_MAX_ACTION_SET_NAME_SIZE);
        stringCopy(actionSetInfo.localizedActionSetName, "Gameplay default", XR_MAX_LOCALIZED_ACTION_SET_NAME_SIZE);

        result = xrCreateActionSet(m_instance, &actionSetInfo, &m_actionSet);
        if (!xrCheck(m_instance, result, "xrCreateActionSet"))
        {
            return false;
        }

        createAction(OpenXRInputType::LEFT_TRACKPAD_X, XR_ACTION_TYPE_FLOAT_INPUT, "left_trackpad_x");
        createAction(OpenXRInputType::LEFT_TRACKPAD_Y, XR_ACTION_TYPE_FLOAT_INPUT, "left_trackpad_y");
        createAction(OpenXRInputType::LEFT_TRACKPAD_CLICK, XR_ACTION_TYPE_BOOLEAN_INPUT, "left_trackpad_click");
        createAction(OpenXRInputType::RIGHT_TRACKPAD_X, XR_ACTION_TYPE_FLOAT_INPUT, "right_trackpad_x");
        createAction(OpenXRInputType::RIGHT_TRACKPAD_Y, XR_ACTION_TYPE_FLOAT_INPUT, "right_trackpad_y");
        createAction(OpenXRInputType::RIGHT_TRACKPAD_CLICK, XR_ACTION_TYPE_BOOLEAN_INPUT, "right_trackpad_click");
        createAction(OpenXRInputType::LEFT_THUMBSTICK_X, XR_ACTION_TYPE_FLOAT_INPUT, "left_thumbstick_x");
        createAction(OpenXRInputType::LEFT_THUMBSTICK_Y, XR_ACTION_TYPE_FLOAT_INPUT, "left_thumbstick_y");
        createAction(OpenXRInputType::LEFT_THUMBSTICK_CLICK, XR_ACTION_TYPE_BOOLEAN_INPUT, "left_thumbstick_click");
        createAction(OpenXRInputType::RIGHT_THUMBSTICK_X, XR_ACTION_TYPE_FLOAT_INPUT, "right_thumbstick_x");
        createAction(OpenXRInputType::RIGHT_THUMBSTICK_Y, XR_ACTION_TYPE_FLOAT_INPUT, "right_thumbstick_y");
        createAction(OpenXRInputType::RIGHT_THUMBSTICK_CLICK, XR_ACTION_TYPE_BOOLEAN_INPUT, "right_thumbstick_click");
        createAction(OpenXRInputType::LEFT_TRIGGER, XR_ACTION_TYPE_FLOAT_INPUT, "left_trigger");
        createAction(OpenXRInputType::RIGHT_TRIGGER, XR_ACTION_TYPE_FLOAT_INPUT, "right_trigger");
        createAction(OpenXRInputType::LEFT_AIM_POSE, XR_ACTION_TYPE_POSE_INPUT, "left_aim_pose");
        createAction(OpenXRInputType::RIGHT_AIM_POSE, XR_ACTION_TYPE_POSE_INPUT, "right_aim_pose");

        return true;
    }

    void OpenXRInput::registerInteractionProfile(const IInteractionProfile &profile)
    {
        XrPath interactionProfilePath;
        XrResult result = xrStringToPath(m_instance, profile.getName().c_str(), &interactionProfilePath);
        if (!xrCheck(m_instance, result, "failed to get interaction profile"))
        {
            return;
        }

        std::vector<XrActionSuggestedBinding> xrActionBindings;
        for (const auto &binding : profile.getBindings())
        {
            XrAction xrAction = getXrAction(binding.first);
            if (!xrAction)
            {
                SIBR_LOG << "Cannot register binding '" << binding.second << "' for profile '" << profile.getName() << "': invalid XRAction" << std::endl;
            }
            else
            {
                XrActionSuggestedBinding actionSuggestedBinding;
                actionSuggestedBinding.action = xrAction;
                actionSuggestedBinding.binding = getXrPath(binding.second);
                xrActionBindings.push_back(actionSuggestedBinding);
            }
        }

        if (xrActionBindings.empty())
        {
            SIBR_WRG << "Invalid action bindings for profile '" << profile.getName() << "': no binding defined!" << std::endl;
            return;
        }

        XrInteractionProfileSuggestedBinding profileSuggestedBindings;
        profileSuggestedBindings.type = XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING;
        profileSuggestedBindings.next = NULL;
        profileSuggestedBindings.interactionProfile = interactionProfilePath;
        profileSuggestedBindings.countSuggestedBindings = (uint32_t)xrActionBindings.size();
        profileSuggestedBindings.suggestedBindings = xrActionBindings.data();

        result = xrSuggestInteractionProfileBindings(m_instance, &profileSuggestedBindings);
        if (!xrCheck(m_instance, result, "failed to suggest bindings"))
        {
            return;
        }
    }

    bool OpenXRInput::attachActionSet()
    {
        XrSessionActionSetsAttachInfo actionSetAttachInfo;
        actionSetAttachInfo.type = XR_TYPE_SESSION_ACTION_SETS_ATTACH_INFO;
        actionSetAttachInfo.next = NULL;
        actionSetAttachInfo.countActionSets = 1;
        actionSetAttachInfo.actionSets = &m_actionSet;
        XrResult result = xrAttachSessionActionSets(m_session, &actionSetAttachInfo);
        return xrCheck(m_instance, result, "failed to attach action set");
    }

    bool OpenXRInput::locateSpaces(XrTime time) {
        for (const auto& space: m_xrSpaces)
        {
            XrAction action = getXrAction(space.first);
            if (!action) {
                continue;
            }

            XrActionStatePose xrValue;
            xrValue.type = XR_TYPE_ACTION_STATE_POSE;
            xrValue.next = NULL;
            XrActionStateGetInfo getInfo = {};
            getInfo.type = XR_TYPE_ACTION_STATE_GET_INFO;
            getInfo.next = NULL;
            getInfo.action = action;
            getInfo.subactionPath = 0;

            XrResult result = xrGetActionStatePose(m_session, &getInfo, &xrValue);
            if (!xrCheck(m_instance, result, "failed to get track pose value!")) {
                continue;
            }

            if (xrValue.isActive)
            {
                XrSpaceVelocity velocity;
                velocity.type = XR_TYPE_SPACE_VELOCITY;
                velocity.next = NULL;
                XrSpaceLocation location;
                location.type = XR_TYPE_SPACE_LOCATION;
                location.next = &velocity;
                XrResult result = xrLocateSpace(space.second, m_refSpace, time, &location);
                if (!xrCheck(m_instance, result, "failed to locate action space"))
                {
                    continue;
                }

                if (location.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT &&
                    location.locationFlags & XR_SPACE_LOCATION_POSITION_TRACKED_BIT &&
                    location.locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT &&
                    location.locationFlags & XR_SPACE_LOCATION_ORIENTATION_TRACKED_BIT)
                {
                    if (space.first == OpenXRInputType::LEFT_AIM_POSE) {
                        m_handPose[static_cast<int>(Hand::LEFT)] = location.pose;
                    } else if (space.first == OpenXRInputType::RIGHT_AIM_POSE) {
                        m_handPose[static_cast<int>(Hand::RIGHT)] = location.pose;
                    }
                    // else: other pose types are not supported
                }
            }
        }
        return true;
    }

    XrPath OpenXRInput::getXrPath(const std::string &path)
    {
        auto it = m_xrPaths.find(path);
        if (it != m_xrPaths.cend())
        {
            return (*it).second;
        }

        XrPath xrPath = XR_NULL_PATH;
        xrStringToPath(m_instance, path.c_str(), &xrPath);
        m_xrPaths.insert({path, xrPath});
        return xrPath;
    }

    XrAction OpenXRInput::getXrAction(OpenXRInputType inputType) const
    {
        auto it = m_xrActions.find(inputType);
        if (it != m_xrActions.cend())
        {
            return (*it).second;
        }
        return XR_NULL_HANDLE;
    }

    bool OpenXRInput::createAction(OpenXRInputType inputType, XrActionType xrActionType, const std::string &name)
    {
        XrAction action;
        XrActionCreateInfo actionInfo;
        actionInfo.type = XR_TYPE_ACTION_CREATE_INFO;
        actionInfo.next = NULL;
        actionInfo.actionType = xrActionType;
        actionInfo.countSubactionPaths = 0;
        actionInfo.subactionPaths = NULL;
        stringCopy(actionInfo.actionName, name.c_str(), XR_MAX_ACTION_NAME_SIZE);
        stringCopy(actionInfo.localizedActionName, name.c_str(), XR_MAX_LOCALIZED_ACTION_NAME_SIZE);

        XrResult result = xrCreateAction(m_actionSet, &actionInfo, &action);
        if (!xrCheck(m_instance, result, "failed to create action for path '%s'", name))
        {
            return false;
        }

        if (xrActionType == XR_ACTION_TYPE_POSE_INPUT)
        {
            XrSpace space = XR_NULL_HANDLE;
            XrActionSpaceCreateInfo aspci = {};
            aspci.type = XR_TYPE_ACTION_SPACE_CREATE_INFO;
            aspci.next = NULL;
            aspci.action = action;
            XrPosef identity;
            identity.orientation = {0.0f, 0.0f, 0.0f, 1.0f};
            identity.position = {0.0f, 0.0f, 0.0f};
            aspci.poseInActionSpace = identity;
            aspci.subactionPath = 0;
            result = xrCreateActionSpace(m_session, &aspci, &space);
            if (!xrCheck(m_instance, result, "xrCreateActionSpace"))
            {
                return false;
            }

            m_xrSpaces.insert({inputType, space});
        }

        m_xrActions.insert({inputType, action});

        return true;
    }

    bool OpenXRInput::extractFloatValue(XrAction xrAction, float &val) const
    {
        if (!xrAction)
        {
            return false;
        }
        XrActionStateFloat xrValue;
        xrValue.type = XR_TYPE_ACTION_STATE_FLOAT;
        xrValue.next = NULL;
        XrActionStateGetInfo getInfo;
        getInfo.type = XR_TYPE_ACTION_STATE_GET_INFO;
        getInfo.next = NULL;
        getInfo.action = xrAction;
        getInfo.subactionPath = 0;

        XrResult result = xrGetActionStateFloat(m_session, &getInfo, &xrValue);
        xrCheck(m_instance, result, "failed to get track value!");

        if (xrValue.isActive)
        {
            val = xrValue.currentState;
            return true;
        }
        return false;
    }

    bool OpenXRInput::extractBooleanValue(XrAction xrAction, bool &val) const
    {
        if (!xrAction)
        {
            return false;
        }
        XrActionStateBoolean xrValue;
        xrValue.type = XR_TYPE_ACTION_STATE_BOOLEAN;
        xrValue.next = NULL;
        XrActionStateGetInfo getInfo;
        getInfo.type = XR_TYPE_ACTION_STATE_GET_INFO;
        getInfo.next = NULL;
        getInfo.action = xrAction;
        getInfo.subactionPath = 0;

        XrResult result = xrGetActionStateBoolean(m_session, &getInfo, &xrValue);
        xrCheck(m_instance, result, "failed to get track value!");

        if (xrValue.isActive && xrValue.changedSinceLastSync)
        {
            val = xrValue.currentState;
            return true;
        }
        return false;
    }


    void OpenXRInput::handleTrackpadEvent(Hand hand)
    {
        int handIdx = static_cast<int>(hand);
        auto clickType = hand == Hand::LEFT ? OpenXRInputType::LEFT_TRACKPAD_CLICK : OpenXRInputType::RIGHT_TRACKPAD_CLICK;
        auto padXType = hand == Hand::LEFT ? OpenXRInputType::LEFT_TRACKPAD_X : OpenXRInputType::RIGHT_TRACKPAD_X;
        auto padYType = hand == Hand::LEFT ? OpenXRInputType::LEFT_TRACKPAD_Y : OpenXRInputType::RIGHT_TRACKPAD_Y;

        bool click = false;
        if (extractBooleanValue(getXrAction(clickType), click))
        {
            if (click)
            {
                if (m_stickClickCallback[handIdx])
                {
                    m_stickClickCallback[handIdx]();
                }

                // HTC Vive controller keeps sending trackpad move events when idle. Let's filter the trackpad move
                // with a click event
                float x, y;
                if (extractFloatValue(getXrAction(padXType), x) &&
                    extractFloatValue(getXrAction(padYType), y))
                {
                    if (m_stickMoveCallback[handIdx])
                    {
                        m_stickMoveCallback[handIdx](x, y);
                    }
                }
            }
        }
    }

    void OpenXRInput::handleThumbstickEvent(Hand hand)
    {
        int handIdx = static_cast<int>(hand);
        auto stickXType = hand == Hand::LEFT ? OpenXRInputType::LEFT_THUMBSTICK_X : OpenXRInputType::RIGHT_THUMBSTICK_X;
        auto stickYType = hand == Hand::LEFT ? OpenXRInputType::LEFT_THUMBSTICK_Y : OpenXRInputType::RIGHT_THUMBSTICK_Y;
        float x, y;
        if (extractFloatValue(getXrAction(stickXType), x) &&
            extractFloatValue(getXrAction(stickYType), y))
        {
            if (m_stickMoveCallback[handIdx])
            {
                m_stickMoveCallback[handIdx](x, y);
            }
        }

        auto clickType = hand == Hand::LEFT ? OpenXRInputType::LEFT_THUMBSTICK_CLICK : OpenXRInputType::RIGHT_TRACKPAD_CLICK;
        bool click = false;
        if (extractBooleanValue(getXrAction(clickType), click))
        {
            if (click && m_stickClickCallback[handIdx])
            {
                m_stickClickCallback[handIdx]();
            }
        }
    }

    void OpenXRInput::handleTriggerEvent(Hand hand)
    {
        int handIdx = static_cast<int>(hand);
        auto triggerType = hand == Hand::LEFT ? OpenXRInputType::LEFT_TRIGGER : OpenXRInputType::RIGHT_TRIGGER;
        float trigger = 0.0f;
        if (extractFloatValue(getXrAction(triggerType), trigger))
        {
            if (m_triggerCallback[handIdx])
            {
                m_triggerCallback[handIdx](trigger);
            }
        }
    }
}
