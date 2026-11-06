//! zestengine-xr: OpenXR bootstrap for the engine (editor + game).
//!
//! This is the *start* of the VR path: it loads the OpenXR loader at runtime,
//! creates an instance with the Vulkan-enable2 extension (so a session can later
//! share zestengine-render's `VkDevice`), and queries the HMD system + its tracking
//! capabilities. It deliberately does NOT yet create the session/swapchains or
//! the frame loop — those plug in once the renderer hands over its Vulkan
//! handles. The point is that both the editor and a shipped game can call
//! [`XrContext::start`] to detect/initialise VR without any of that wired yet.
//!
//! Runtime targets: Monado and SteamVR's OpenXR runtime (SteamVR exposes Vive
//! trackers through OpenXR tracker extensions, which the full-body path will use
//! to feed `zestengine_core::TrackerTargets` into the IK solver), and on Android the
//! headset's system runtime (Meta Quest) through the Khronos loader shipped in
//! the APK (`lib/arm64-v8a/libopenxr_loader.so`, see tools/android/build_apk.sh).

use anyhow::{Context as _, Result};

/// A live OpenXR instance + selected HMD system. Held by the engine for the app
/// lifetime; dropping it tears the instance down.
pub struct XrContext {
    /// The OpenXR entry (loader). Kept alive for the instance's lifetime.
    pub entry: openxr::Entry,
    pub instance: openxr::Instance,
    pub system: openxr::SystemId,
    pub system_name: String,
    pub orientation_tracking: bool,
    pub position_tracking: bool,
    /// `XR_KHR_convert_timespec_time` available → `Instance::now()` is safe.
    time_supported: bool,
    /// `XR_HTCX_vive_tracker_interaction` available → suggest tracker bindings.
    tracker_supported: bool,
    /// `XR_FB_face_tracking2` (WiVRn, Virtual Desktop, Quest Pro / 3).
    face_supported: bool,
    /// `XR_EXT_eye_gaze_interaction`.
    eye_gaze_supported: bool,
    /// `XR_EXT_hand_tracking` (Quest hands; WiVRn / Virtual Desktop / Monado
    /// forward it).
    hand_tracking_supported: bool,
    /// `XR_FB_display_refresh_rate` (Quest: 72 / 80 / 90 / 120 Hz).
    refresh_rate_supported: bool,
}

/// Why VR is switched off for every process on this machine, if it is:
/// `ZESTENGINE_VR=0|false|off|no`, or a `zestengine-no-vr` marker file in
/// `$XDG_RUNTIME_DIR` (else `/tmp`). Neither a project's "Start in VR" nor a
/// caller can override it; delete the file / unset the variable to allow VR again.
pub fn vr_disabled_globally() -> Option<String> {
    let markers: Vec<std::path::PathBuf> = std::env::var_os("XDG_RUNTIME_DIR")
        .map(|d| std::path::PathBuf::from(d).join("zestengine-no-vr"))
        .into_iter()
        .chain(std::iter::once(std::path::PathBuf::from("/tmp/zestengine-no-vr")))
        .collect();
    vr_disabled_by(std::env::var("ZESTENGINE_VR").ok().as_deref(), &markers)
}

fn vr_disabled_by(env: Option<&str>, markers: &[std::path::PathBuf]) -> Option<String> {
    if let Some(v) = env
        && matches!(v.trim().to_ascii_lowercase().as_str(), "0" | "false" | "off" | "no")
    {
        return Some("ZESTENGINE_VR is off".into());
    }
    markers.iter().find(|p| p.is_file()).map(|p| format!("{} exists (delete it to allow VR)", p.display()))
}

#[cfg(test)]
mod no_vr_tests {
    use super::vr_disabled_by;

    #[test]
    fn env_and_marker_switch_vr_off() {
        let dir = std::env::temp_dir().join(format!("zestengine-no-vr-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let marker = dir.join("zestengine-no-vr");
        assert!(vr_disabled_by(None, std::slice::from_ref(&marker)).is_none(), "no env, no file: VR allowed");
        assert!(vr_disabled_by(Some("1"), std::slice::from_ref(&marker)).is_none(), "env on: allowed");
        assert!(vr_disabled_by(Some(" OFF "), &[]).is_some(), "env off wins without any file");
        std::fs::write(&marker, "x").unwrap();
        assert!(vr_disabled_by(None, std::slice::from_ref(&marker)).is_some(), "marker file switches VR off");
        assert!(vr_disabled_by(Some("1"), std::slice::from_ref(&marker)).is_some(), "the marker is not overridable by the env");
        let _ = std::fs::remove_dir_all(&dir);
    }
}

impl XrContext {
    /// Try to start OpenXR: load the loader, create an instance (Vulkan2), and
    /// select the HMD system. Returns `Ok(None)` when VR is simply unavailable
    /// (no runtime installed, or no headset connected) so the caller can carry
    /// on flat; returns `Err` only on an unexpected runtime fault.
    pub fn start(app_name: &str) -> Result<Option<Self>> {
        // Machine-wide off switch, ahead of every project / caller setting: while
        // another VR application owns the headset (or a headless test run must not
        // touch the runtime), `ZESTENGINE_VR=0` or an existing
        // `$XDG_RUNTIME_DIR/zestengine-no-vr` (else `/tmp/zestengine-no-vr`) keeps
        // every editor, player and test flat without opening the OpenXR runtime.
        if let Some(why) = vr_disabled_globally() {
            tracing::info!("VR disabled: {why}");
            return Ok(None);
        }
        // Load the OpenXR loader at runtime (dlopen) rather than link-time, so a
        // build/run without a runtime installed degrades gracefully.
        let entry = match unsafe { openxr::Entry::load() } {
            Ok(e) => e,
            Err(e) => {
                tracing::info!("OpenXR loader not found, VR disabled: {e}");
                return Ok(None);
            }
        };
        // Android: the Khronos loader (libopenxr_loader.so in the APK) needs the
        // JavaVM + activity before any other call, to find the system runtime.
        #[cfg(target_os = "android")]
        if let Err(e) = entry.initialize_android_loader() {
            tracing::warn!("xrInitializeLoaderKHR failed, VR disabled: {e}");
            return Ok(None);
        }

        let available = entry
            .enumerate_extensions()
            .context("enumerating OpenXR extensions")?;
        if !available.khr_vulkan_enable2 {
            tracing::warn!("OpenXR runtime lacks XR_KHR_vulkan_enable2; VR disabled");
            return Ok(None);
        }

        let mut extensions = openxr::ExtensionSet::default();
        extensions.khr_vulkan_enable2 = true;
        // Android runtimes (Quest) take the JavaVM + activity at instance
        // creation; the openxr crate chains them in when this is set.
        #[cfg(target_os = "android")]
        {
            if !available.khr_android_create_instance {
                tracing::warn!("OpenXR runtime lacks XR_KHR_android_create_instance; VR disabled");
                return Ok(None);
            }
            extensions.khr_android_create_instance = true;
        }
        // Instance::now() (used to timestamp head/tracker pose lookups) needs the
        // timespec-conversion extension on Linux runtimes; enabling it avoids the
        // `KHR_convert_timespec_time not loaded` panic. Tracked so the pose readers
        // can fall back gracefully if a runtime lacks it.
        let time_supported = available.khr_convert_timespec_time;
        if time_supported {
            extensions.khr_convert_timespec_time = true;
        }
        // SteamVR/Vive full-body trackers (waist/feet) come through this
        // extension; enable it when the runtime offers it.
        let tracker_supported = available.htcx_vive_tracker_interaction;
        if tracker_supported {
            extensions.htcx_vive_tracker_interaction = true;
        }
        // Face and eye tracking: used when the runtime offers them.
        let face_supported = available.fb_face_tracking2;
        extensions.fb_face_tracking2 = face_supported;
        let eye_gaze_supported = available.ext_eye_gaze_interaction;
        extensions.ext_eye_gaze_interaction = eye_gaze_supported;
        let hand_tracking_supported = available.ext_hand_tracking;
        extensions.ext_hand_tracking = hand_tracking_supported;
        let refresh_rate_supported = available.fb_display_refresh_rate;
        extensions.fb_display_refresh_rate = refresh_rate_supported;

        let app_info = openxr::ApplicationInfo {
            application_name: app_name,
            application_version: 1,
            engine_name: "zestengine",
            engine_version: 1,
            api_version: openxr::Version::new(1, 0, 0),
        };
        let instance = entry
            .create_instance(&app_info, &extensions, &[])
            .context("creating OpenXR instance")?;

        // Select the head-mounted display. FORM_FACTOR_UNAVAILABLE means no HMD
        // is connected/active — not an error, just "no VR right now".
        let system = match instance.system(openxr::FormFactor::HEAD_MOUNTED_DISPLAY) {
            Ok(s) => s,
            Err(openxr::sys::Result::ERROR_FORM_FACTOR_UNAVAILABLE) => {
                tracing::info!("No OpenXR HMD connected; VR disabled");
                return Ok(None);
            }
            Err(e) => return Err(anyhow::anyhow!("selecting OpenXR system: {e}")),
        };

        let props = instance
            .system_properties(system)
            .context("querying OpenXR system properties")?;

        tracing::info!(
            system = %props.system_name,
            orientation = props.tracking_properties.orientation_tracking,
            position = props.tracking_properties.position_tracking,
            "OpenXR started"
        );

        Ok(Some(Self {
            entry,
            instance,
            system,
            system_name: props.system_name.clone(),
            orientation_tracking: props.tracking_properties.orientation_tracking,
            position_tracking: props.tracking_properties.position_tracking,
            time_supported,
            tracker_supported,
            face_supported,
            eye_gaze_supported,
            hand_tracking_supported,
            refresh_rate_supported,
        }))
    }

    /// Human-readable HMD/runtime name (for the editor status line / logs).
    pub fn system_name(&self) -> &str {
        &self.system_name
    }

    /// Create the VR session, sharing zestengine-render's Vulkan device
    /// (`XR_KHR_vulkan_enable2`). Pass the renderer's raw handles
    /// (`Renderer::vulkan_raw_handles`). The session must be polled each frame
    /// ([`XrSession::poll`]) to drive its lifecycle, after which head/controller
    /// poses can be read ([`XrSession::head_pose`]).
    pub fn create_session(
        &self,
        vk_instance: u64,
        vk_physical_device: u64,
        vk_device: u64,
        queue_family_index: u32,
    ) -> Result<XrSession> {
        // Spec requires querying graphics requirements before session creation.
        let _ = self
            .instance
            .graphics_requirements::<openxr::Vulkan>(self.system)
            .context("OpenXR Vulkan graphics requirements")?;
        // With XR_KHR_vulkan_enable2 the runtime MANDATES a call to
        // xrGetVulkanGraphicsDevice2KHR before xrCreateSession (it validates both
        // that the call happened AND that the session uses the device it returns —
        // the HMD's GPU). Skipping it is the `XR_ERROR_VALIDATION_FAILURE: Has not
        // called xrGetVulkanGraphicsDevice2KHR` seen on WiVRn/Monado/SteamVR.
        let vk_instance_ptr = vk_instance as usize as *const std::ffi::c_void;
        let xr_physical_device = unsafe {
            self.instance
                .vulkan_graphics_device(self.system, vk_instance_ptr)
        }
        .context("xrGetVulkanGraphicsDevice2KHR")?;
        // On a single-GPU machine this equals the renderer's pick. If it differs
        // (multi-GPU laptop, HMD on the dGPU), the renderer's VkDevice was built on
        // the wrong GPU and the session will still fail — the real fix is to create
        // the renderer on this device. Warn so that's diagnosable.
        let our_physical_device = vk_physical_device as usize as *const std::ffi::c_void;
        if xr_physical_device != our_physical_device {
            tracing::warn!(
                "OpenXR requires a different Vulkan physical device than the renderer \
                 selected (multi-GPU?). The renderer must be created on the HMD's GPU; \
                 the session create below will likely fail until then."
            );
        }
        let info = openxr::vulkan::SessionCreateInfo {
            instance: vk_instance_ptr,
            physical_device: xr_physical_device,
            device: vk_device as usize as *const std::ffi::c_void,
            queue_family_index,
            queue_index: 0,
        };
        let (session, frame_waiter, frame_stream) = unsafe {
            self.instance
                .create_session::<openxr::Vulkan>(self.system, &info)
                .context("creating OpenXR session")?
        };
        let identity = openxr::Posef {
            orientation: openxr::Quaternionf {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                w: 1.0,
            },
            position: openxr::Vector3f {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
        };
        // STAGE needs a configured play area; standalone headsets with a
        // stationary boundary may only offer LOCAL, which still tracks.
        let stage = session
            .create_reference_space(openxr::ReferenceSpaceType::STAGE, identity)
            .or_else(|e| {
                tracing::warn!("OpenXR STAGE space unavailable ({e}); using LOCAL");
                session.create_reference_space(openxr::ReferenceSpaceType::LOCAL, identity)
            })?;
        let view = session.create_reference_space(openxr::ReferenceSpaceType::VIEW, identity)?;

        // Full-body tracker action set: pose actions for hands (controllers) +
        // waist/feet (SteamVR/Vive trackers), each with an action space located
        // against the stage every frame.
        let action_set = self
            .instance
            .create_action_set("fullbody", "Full Body Tracking", 0)
            .context("creating OpenXR action set")?;
        let mk = |name: &str, label: &str| -> Result<openxr::Action<openxr::Posef>> {
            action_set
                .create_action::<openxr::Posef>(name, label, &[])
                .map_err(|e| anyhow::anyhow!("creating action {name}: {e}"))
        };
        let a_left_hand = mk("left_hand", "Left Hand")?;
        let a_right_hand = mk("right_hand", "Right Hand")?;
        let a_hips = mk("hips", "Hips")?;
        let a_left_foot = mk("left_foot", "Left Foot")?;
        let a_right_foot = mk("right_foot", "Right Foot")?;
        let a_chest = mk("chest", "Chest")?;
        let a_left_elbow = mk("left_elbow", "Left Elbow")?;
        let a_right_elbow = mk("right_elbow", "Right Elbow")?;
        let a_left_knee = mk("left_knee", "Left Knee")?;
        let a_right_knee = mk("right_knee", "Right Knee")?;
        let a_eye_gaze = mk("eye_gaze", "Eye Gaze")?;

        // Locomotion / interaction inputs (analog sticks + grips, digital trigger +
        // menu button). Bound to the Oculus Touch profile below (the common case);
        // a runtime with a different controller just ignores the suggestions.
        let mkf = |name: &str, label: &str| -> Result<openxr::Action<f32>> {
            action_set
                .create_action::<f32>(name, label, &[])
                .map_err(|e| anyhow::anyhow!("creating action {name}: {e}"))
        };
        let mkb = |name: &str, label: &str| -> Result<openxr::Action<bool>> {
            action_set
                .create_action::<bool>(name, label, &[])
                .map_err(|e| anyhow::anyhow!("creating action {name}: {e}"))
        };
        let a_lstick_x = mkf("left_stick_x", "Left Stick X")?;
        let a_lstick_y = mkf("left_stick_y", "Left Stick Y")?;
        let a_rstick_x = mkf("right_stick_x", "Right Stick X")?;
        let a_rstick_y = mkf("right_stick_y", "Right Stick Y")?;
        let a_lgrip = mkf("left_grip", "Left Grip")?;
        let a_rgrip = mkf("right_grip", "Right Grip")?;
        let a_rtrigger = mkb("right_trigger", "Right Trigger")?;
        let a_ltrigger = mkb("left_trigger", "Left Trigger")?;
        let a_ltrigger_value = mkf("left_trigger_value", "Left Trigger Pull")?;
        let a_rtrigger_value = mkf("right_trigger_value", "Right Trigger Pull")?;
        let a_a = mkb("button_a", "A")?;
        let a_b = mkb("button_b", "B")?;
        let a_x = mkb("button_x", "X")?;
        let a_y = mkb("button_y", "Y")?;
        let a_lstick_click = mkb("left_stick_click", "Left Stick Click")?;
        let a_rstick_click = mkb("right_stick_click", "Right Stick Click")?;
        let a_menu = mkb("menu", "Menu")?;

        // Suggested bindings. Hands map to the cross-vendor simple controller;
        // waist/feet to Vive tracker role paths (best-effort — ignored if the
        // runtime lacks the tracker extension). Failures here are non-fatal.
        let path = |s: &str| self.instance.string_to_path(s);
        if let (Ok(lh), Ok(rh)) = (
            path("/user/hand/left/input/aim/pose"),
            path("/user/hand/right/input/aim/pose"),
        ) {
            let _ = self.instance.suggest_interaction_profile_bindings(
                path("/interaction_profiles/khr/simple_controller").unwrap_or(openxr::Path::NULL),
                &[
                    openxr::Binding::new(&a_left_hand, lh),
                    openxr::Binding::new(&a_right_hand, rh),
                ],
            );
        }
        if self.tracker_supported {
            let roles: [(&str, &openxr::Action<openxr::Posef>); 8] = [
                ("waist", &a_hips),
                ("left_foot", &a_left_foot),
                ("right_foot", &a_right_foot),
                ("chest", &a_chest),
                ("left_elbow", &a_left_elbow),
                ("right_elbow", &a_right_elbow),
                ("left_knee", &a_left_knee),
                ("right_knee", &a_right_knee),
            ];
            let bindings: Vec<openxr::Binding> = roles
                .iter()
                .filter_map(|(role, action)| {
                    let p = path(&format!("/user/vive_tracker_htcx/role/{role}/input/grip/pose")).ok()?;
                    Some(openxr::Binding::new(*action, p))
                })
                .collect();
            let _ = self.instance.suggest_interaction_profile_bindings(
                path("/interaction_profiles/htc/vive_tracker_htcx").unwrap_or(openxr::Path::NULL),
                &bindings,
            );
        }
        if self.eye_gaze_supported
            && let (Ok(profile), Ok(gaze)) =
                (path("/interaction_profiles/ext/eye_gaze_interaction"), path("/user/eyes_ext/input/gaze_ext/pose"))
        {
            let _ = self.instance.suggest_interaction_profile_bindings(profile, &[openxr::Binding::new(&a_eye_gaze, gaze)]);
        }
        // Locomotion/interaction bindings on the Oculus Touch profile (Quest, Rift;
        // the most common runtime). Pose hands also bind here so controllers track.
        // Touch Plus (Quest 3 / 3S) and Touch Pro also drive this profile, so
        // the face buttons, analog trigger pull and stick clicks bind here too.
        let touch_extra = [
            ("/user/hand/left/input/trigger/value", &a_ltrigger_value),
            ("/user/hand/right/input/trigger/value", &a_rtrigger_value),
        ];
        let touch_buttons = [
            ("/user/hand/right/input/a/click", &a_a),
            ("/user/hand/right/input/b/click", &a_b),
            ("/user/hand/left/input/x/click", &a_x),
            ("/user/hand/left/input/y/click", &a_y),
            ("/user/hand/left/input/thumbstick/click", &a_lstick_click),
            ("/user/hand/right/input/thumbstick/click", &a_rstick_click),
        ];
        if let Ok(profile) = path("/interaction_profiles/oculus/touch_controller") {
            let base: [(&str, BindingAction); 11] = [
                ("/user/hand/left/input/aim/pose", BindingAction::Pose(&a_left_hand)),
                ("/user/hand/right/input/aim/pose", BindingAction::Pose(&a_right_hand)),
                ("/user/hand/left/input/thumbstick/x", BindingAction::Float(&a_lstick_x)),
                ("/user/hand/left/input/thumbstick/y", BindingAction::Float(&a_lstick_y)),
                ("/user/hand/right/input/thumbstick/x", BindingAction::Float(&a_rstick_x)),
                ("/user/hand/right/input/thumbstick/y", BindingAction::Float(&a_rstick_y)),
                ("/user/hand/left/input/squeeze/value", BindingAction::Float(&a_lgrip)),
                ("/user/hand/right/input/squeeze/value", BindingAction::Float(&a_rgrip)),
                ("/user/hand/right/input/trigger/value", BindingAction::Bool(&a_rtrigger)),
                ("/user/hand/left/input/trigger/value", BindingAction::Bool(&a_ltrigger)),
                ("/user/hand/left/input/y/click", BindingAction::Bool(&a_menu)),
            ];
            let mut bindings = Vec::new();
            for (p, action) in base {
                let Ok(p) = path(p) else { continue };
                bindings.push(match action {
                    BindingAction::Pose(a) => openxr::Binding::new(a, p),
                    BindingAction::Float(a) => openxr::Binding::new(a, p),
                    BindingAction::Bool(a) => openxr::Binding::new(a, p),
                });
            }
            for (p, a) in touch_extra {
                if let Ok(p) = path(p) {
                    bindings.push(openxr::Binding::new(a, p));
                }
            }
            for (p, a) in touch_buttons {
                if let Ok(p) = path(p) {
                    bindings.push(openxr::Binding::new(a, p));
                }
            }
            if let Err(e) = self.instance.suggest_interaction_profile_bindings(profile, &bindings) {
                tracing::warn!("OpenXR Touch controller bindings rejected: {e}");
            }
        }
        session
            .attach_action_sets(&[&action_set])
            .context("attaching OpenXR action set")?;

        let space = |a: &openxr::Action<openxr::Posef>| a.create_space(&session, openxr::Path::NULL, identity);
        let left_hand = space(&a_left_hand)?;
        let right_hand = space(&a_right_hand)?;
        let hips = space(&a_hips)?;
        let left_foot = space(&a_left_foot)?;
        let right_foot = space(&a_right_foot)?;
        let chest = space(&a_chest)?;
        let left_elbow = space(&a_left_elbow)?;
        let right_elbow = space(&a_right_elbow)?;
        let left_knee = space(&a_left_knee)?;
        let right_knee = space(&a_right_knee)?;
        let eye_gaze = if self.eye_gaze_supported { space(&a_eye_gaze).ok() } else { None };
        let face_tracker = if self.face_supported { create_face_tracker(&self.instance, &session) } else { None };
        tracing::info!(face = face_tracker.is_some(), eye_gaze = eye_gaze.is_some(), "OpenXR face / eye tracking");
        // Hand tracking (XR_EXT_hand_tracking): per-hand joint skeletons, drawn
        // by the editor overlay in edit mode. Optional — controllers still work.
        let hand_trackers = if self.hand_tracking_supported {
            let tracker = |hand| {
                session.create_hand_tracker(hand).map_err(|e| {
                    tracing::warn!("OpenXR {hand:?} hand tracker unavailable: {e}");
                }).ok()
            };
            [tracker(openxr::Hand::LEFT), tracker(openxr::Hand::RIGHT)]
        } else {
            [None, None]
        };
        tracing::info!(hand_tracking = hand_trackers[0].is_some(), "OpenXR hand tracking");

        Ok(XrSession {
            instance: self.instance.clone(),
            system: self.system,
            session,
            frame_waiter,
            frame_stream,
            stage,
            view,
            action_set,
            left_hand,
            right_hand,
            hips,
            left_foot,
            right_foot,
            chest,
            left_elbow,
            right_elbow,
            left_knee,
            right_knee,
            eye_gaze,
            face_tracker,
            a_lstick_x,
            a_lstick_y,
            a_rstick_x,
            a_rstick_y,
            a_lgrip,
            a_rgrip,
            a_rtrigger,
            a_ltrigger,
            a_ltrigger_value,
            a_rtrigger_value,
            a_a,
            a_b,
            a_x,
            a_y,
            a_lstick_click,
            a_rstick_click,
            a_menu,
            hand_trackers,
            running: false,
            lifecycle: SessionLifecycle::default(),
            last_display_time: None,
            swapchains: Vec::new(),
            swap_images: Vec::new(),
            view_extent: (0, 0),
            multiview: false,
            blend_mode: openxr::EnvironmentBlendMode::OPAQUE,
            time_supported: self.time_supported,
            refresh_rate_supported: self.refresh_rate_supported,
        })
    }
}

/// Controller inputs read this frame (analog sticks/grips + digital trigger/menu).
/// Bound to the Oculus Touch profile; zero/false when unbound or not tracking.
#[derive(Clone, Copy, Debug, Default)]
pub struct VrInput {
    pub left_stick: (f32, f32),
    pub right_stick: (f32, f32),
    pub left_grip: f32,
    pub right_grip: f32,
    pub left_trigger: bool,
    pub right_trigger: bool,
    pub menu: bool,
    /// Analog trigger pull (0..1).
    pub left_trigger_value: f32,
    pub right_trigger_value: f32,
    /// Face buttons: X / Y on the left controller, A / B on the right.
    pub a: bool,
    pub b: bool,
    pub x: bool,
    pub y: bool,
    pub left_stick_click: bool,
    pub right_stick_click: bool,
}

/// A typed action for building one interaction profile's binding list.
enum BindingAction<'a> {
    Pose(&'a openxr::Action<openxr::Posef>),
    Float(&'a openxr::Action<f32>),
    Bool(&'a openxr::Action<bool>),
}

/// What the app must do in response to an OpenXR session state change.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LifecycleAction {
    None,
    /// READY: call `xrBeginSession`.
    Begin,
    /// STOPPING: call `xrEndSession` and stop submitting frames.
    End,
    /// EXITING / LOSS_PENDING: the runtime wants the app (or session) gone.
    Exit,
}

/// The OpenXR session state machine, kept apart from the runtime so the
/// transitions are testable. On a standalone headset the runtime walks it as
/// the app is paused and resumed (system menu, headset off, app switch):
/// FOCUSED -> VISIBLE -> SYNCHRONIZED -> STOPPING -> IDLE, then back up
/// through READY when the app returns.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SessionLifecycle {
    /// Between a successful `xrBeginSession` and STOPPING.
    pub running: bool,
    /// The compositor shows this app's frames (VISIBLE or FOCUSED).
    pub visible: bool,
    /// The app has input focus (FOCUSED). Without it controller input reads
    /// inactive, e.g. while the system menu is open.
    pub focused: bool,
    /// EXITING or LOSS_PENDING was seen.
    pub exit_requested: bool,
}

impl SessionLifecycle {
    /// Apply a session state change and return what the caller must do.
    /// After [`LifecycleAction::Begin`] the caller reports the outcome with
    /// [`SessionLifecycle::began`].
    pub fn on_state(&mut self, state: openxr::SessionState) -> LifecycleAction {
        match state {
            openxr::SessionState::READY => LifecycleAction::Begin,
            openxr::SessionState::SYNCHRONIZED => {
                self.visible = false;
                self.focused = false;
                LifecycleAction::None
            }
            openxr::SessionState::VISIBLE => {
                self.visible = true;
                self.focused = false;
                LifecycleAction::None
            }
            openxr::SessionState::FOCUSED => {
                self.visible = true;
                self.focused = true;
                LifecycleAction::None
            }
            openxr::SessionState::STOPPING => {
                self.visible = false;
                self.focused = false;
                self.running = false;
                LifecycleAction::End
            }
            openxr::SessionState::EXITING | openxr::SessionState::LOSS_PENDING => {
                self.visible = false;
                self.focused = false;
                self.running = false;
                self.exit_requested = true;
                LifecycleAction::Exit
            }
            _ => LifecycleAction::None,
        }
    }

    /// Record whether `xrBeginSession` succeeded after a READY.
    pub fn began(&mut self, ok: bool) {
        self.running = ok;
    }
}

/// Tracked full-body poses (stage space) read from the VR session this frame.
/// Each is `None` until its device is present + tracking. Maps directly to
/// `zestengine_core::TrackerTargets`.
#[derive(Clone, Copy, Debug, Default)]
pub struct BodyPoses {
    pub head: Option<(glam::Vec3, glam::Quat)>,
    pub left_hand: Option<(glam::Vec3, glam::Quat)>,
    pub right_hand: Option<(glam::Vec3, glam::Quat)>,
    pub hips: Option<(glam::Vec3, glam::Quat)>,
    pub left_foot: Option<(glam::Vec3, glam::Quat)>,
    pub right_foot: Option<(glam::Vec3, glam::Quat)>,
    pub chest: Option<(glam::Vec3, glam::Quat)>,
    pub left_elbow: Option<(glam::Vec3, glam::Quat)>,
    pub right_elbow: Option<(glam::Vec3, glam::Quat)>,
    pub left_knee: Option<(glam::Vec3, glam::Quat)>,
    pub right_knee: Option<(glam::Vec3, glam::Quat)>,
}

/// Tracked hand skeleton (XR_EXT_hand_tracking): per-joint positions + radii in
/// the locate space (the stage), with a validity mask. Index with
/// [`openxr::HandJoint`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HandJoints {
    pub positions: [glam::Vec3; openxr::HAND_JOINT_COUNT],
    pub radii: [f32; openxr::HAND_JOINT_COUNT],
    /// True where the runtime reported a tracked joint position this frame.
    pub valid: [bool; openxr::HAND_JOINT_COUNT],
}

/// A live OpenXR session sharing the renderer's Vulkan device. Drives its own
/// lifecycle via [`poll`] and exposes tracked poses (head now; controllers +
/// SteamVR/Vive trackers via action sets are the next step). Per-eye swapchains
/// + stereo rendering are wired separately once the renderer exposes them.
pub struct XrSession {
    instance: openxr::Instance,
    system: openxr::SystemId,
    pub session: openxr::Session<openxr::Vulkan>,
    frame_waiter: openxr::FrameWaiter,
    frame_stream: openxr::FrameStream<openxr::Vulkan>,
    stage: openxr::Space,
    view: openxr::Space,
    action_set: openxr::ActionSet,
    left_hand: openxr::Space,
    right_hand: openxr::Space,
    hips: openxr::Space,
    left_foot: openxr::Space,
    right_foot: openxr::Space,
    chest: openxr::Space,
    left_elbow: openxr::Space,
    right_elbow: openxr::Space,
    left_knee: openxr::Space,
    right_knee: openxr::Space,
    eye_gaze: Option<openxr::Space>,
    face_tracker: Option<openxr::sys::FaceTracker2FB>,
    a_lstick_x: openxr::Action<f32>,
    a_lstick_y: openxr::Action<f32>,
    a_rstick_x: openxr::Action<f32>,
    a_rstick_y: openxr::Action<f32>,
    a_lgrip: openxr::Action<f32>,
    a_rgrip: openxr::Action<f32>,
    a_rtrigger: openxr::Action<bool>,
    a_ltrigger: openxr::Action<bool>,
    a_ltrigger_value: openxr::Action<f32>,
    a_rtrigger_value: openxr::Action<f32>,
    a_a: openxr::Action<bool>,
    a_b: openxr::Action<bool>,
    a_x: openxr::Action<bool>,
    a_y: openxr::Action<bool>,
    a_lstick_click: openxr::Action<bool>,
    a_rstick_click: openxr::Action<bool>,
    a_menu: openxr::Action<bool>,
    hand_trackers: [Option<openxr::HandTracker>; 2],
    pub running: bool,
    lifecycle: SessionLifecycle,
    /// Predicted display time of the last `xrWaitFrame`: the pose time when
    /// the runtime lacks `XR_KHR_convert_timespec_time`.
    last_display_time: Option<openxr::Time>,
    // Stereo render: per-eye swapchains (or one 2-layer swapchain under
    // multiview) + their VkImage handles, render extent.
    swapchains: Vec<openxr::Swapchain<openxr::Vulkan>>,
    swap_images: Vec<Vec<u64>>,
    view_extent: (u32, u32),
    multiview: bool,
    blend_mode: openxr::EnvironmentBlendMode,
    /// `Instance::now()` is only safe when `XR_KHR_convert_timespec_time` loaded.
    time_supported: bool,
    refresh_rate_supported: bool,
}

/// Raw `VkFormat`s the eye images can use, best first after the caller's own
/// preference: 8-bit sRGB (the post pass writes linear, the hardware encodes),
/// then half float (linear for the compositor).
const XR_FALLBACK_FORMATS: [u32; 3] = [
    43, // VK_FORMAT_R8G8B8A8_SRGB
    50, // VK_FORMAT_B8G8R8A8_SRGB
    97, // VK_FORMAT_R16G16B16A16_SFLOAT
];

/// The eye swapchains [`XrSession::setup_stereo`] made.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StereoSetup {
    /// Per-eye render extent (both eyes share it).
    pub width: u32,
    pub height: u32,
    /// Raw `VkFormat` of the eye images.
    pub format: u32,
    /// One swapchain with an array layer per eye (`XrEye::layer`); render both
    /// eyes in one multiview pass.
    pub multiview: bool,
}

/// First of `preferred`, then [`XR_FALLBACK_FORMATS`], that the runtime offers.
fn pick_swapchain_format(preferred: &[u32], offered: &[u32]) -> Option<u32> {
    preferred.iter().chain(XR_FALLBACK_FORMATS.iter()).copied().find(|f| offered.contains(f))
}

/// One XR frame's per-eye render targets + view/projection (from [`begin_frame`]).
pub struct XrFrame {
    pub display_time: openxr::Time,
    pub eyes: Vec<XrEye>,
}

/// A single eye to render this frame.
pub struct XrEye {
    /// Raw `VkImage` (u64) of the acquired swapchain image to render into.
    pub image: u64,
    /// Array layer of `image` for this eye (non-zero only under multiview,
    /// where both eyes share one 2-layer image).
    pub layer: u32,
    pub image_index: u32,
    /// View matrix (world → eye) and projection (asymmetric, from the eye FOV).
    pub view: glam::Mat4,
    pub proj: glam::Mat4,
    pub pose: openxr::Posef,
    pub fov: openxr::Fovf,
    pub width: u32,
    pub height: u32,
}

impl XrFrame {
    /// One view (stage space) whose frustum encloses every eye's: the eyes'
    /// shared orientation from a point behind them, with the union of their
    /// fields of view. Fits shadow cascades and culls a multiview pass. None
    /// for canted displays (eye orientations differ) or unusual FOVs.
    pub fn combined_view(&self, near: f32, far: f32) -> Option<(glam::Mat4, glam::Mat4)> {
        let poses: Vec<(glam::Vec3, glam::Quat, openxr::Fovf)> = self
            .eyes
            .iter()
            .map(|e| {
                let p = e.pose.position;
                let o = e.pose.orientation;
                (glam::Vec3::new(p.x, p.y, p.z), glam::Quat::from_xyzw(o.x, o.y, o.z, o.w).normalize(), e.fov)
            })
            .collect();
        combined_view(&poses, near, far)
    }
}

/// See [`XrFrame::combined_view`]: eye world poses + FOVs in, (view, proj) out.
fn combined_view(eyes: &[(glam::Vec3, glam::Quat, openxr::Fovf)], near: f32, far: f32) -> Option<(glam::Mat4, glam::Mat4)> {
    let &(_, q, _) = eyes.first()?;
    if eyes.iter().any(|(_, qi, _)| qi.dot(q).abs() < 0.9999) {
        return None;
    }
    let (mut l, mut r, mut u, mut d) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for (_, _, f) in eyes {
        l = l.min(f.angle_left.tan());
        r = r.max(f.angle_right.tan());
        u = u.max(f.angle_up.tan());
        d = d.min(f.angle_down.tan());
    }
    if l >= 0.0 || r <= 0.0 || u <= 0.0 || d >= 0.0 {
        return None;
    }
    let center = eyes.iter().map(|(p, _, _)| *p).sum::<glam::Vec3>() / eyes.len() as f32;
    // Pull the apex back until each eye's side planes sit inside the union.
    let inv = q.inverse();
    let mut back = 0.0f32;
    for (p, _, _) in eyes {
        let local = inv * (*p - center);
        back = back.max(local.x / r).max(local.x / l).max(local.y / u).max(local.y / d).max(local.z);
    }
    let apex = center + q * glam::Vec3::new(0.0, 0.0, back);
    let fov = openxr::Fovf { angle_left: l.atan(), angle_right: r.atan(), angle_up: u.atan(), angle_down: d.atan() };
    let view = glam::Mat4::from_rotation_translation(q, apex).inverse();
    Some((view, fov_to_proj(fov, near, far + back)))
}

/// Asymmetric perspective projection from an OpenXR field of view (Vulkan depth
/// 0..1). Matches the OpenXR projection convention.
fn fov_to_proj(fov: openxr::Fovf, near: f32, far: f32) -> glam::Mat4 {
    let l = fov.angle_left.tan();
    let r = fov.angle_right.tan();
    let u = fov.angle_up.tan();
    let d = fov.angle_down.tan();
    let w = r - l;
    let h = u - d;
    glam::Mat4::from_cols_array(&[
        2.0 / w, 0.0, 0.0, 0.0,
        0.0, 2.0 / h, 0.0, 0.0,
        (r + l) / w, (u + d) / h, -far / (far - near), -1.0,
        0.0, 0.0, -(far * near) / (far - near), 0.0,
    ])
}

/// View (world → eye) matrix = inverse of the eye's world pose.
fn view_to_view_matrix(pose: openxr::Posef) -> glam::Mat4 {
    let q = glam::Quat::from_xyzw(
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    );
    let p = glam::Vec3::new(pose.position.x, pose.position.y, pose.position.z);
    glam::Mat4::from_rotation_translation(q, p).inverse()
}

impl XrSession {
    /// Pump the OpenXR event queue and drive the session lifecycle (begin on
    /// READY, end on STOPPING). Call once per frame.
    pub fn poll(&mut self) {
        let mut buf = openxr::EventDataBuffer::new();
        while let Ok(Some(event)) = self.instance.poll_event(&mut buf) {
            match event {
                openxr::Event::SessionStateChanged(e) => {
                    tracing::debug!(state = ?e.state(), "OpenXR session state");
                    match self.lifecycle.on_state(e.state()) {
                        LifecycleAction::Begin => {
                            let ok = self
                                .session
                                .begin(openxr::ViewConfigurationType::PRIMARY_STEREO)
                                .is_ok();
                            self.lifecycle.began(ok);
                            if ok {
                                tracing::info!("OpenXR session running");
                            }
                        }
                        LifecycleAction::End => {
                            let _ = self.session.end();
                        }
                        LifecycleAction::Exit => tracing::info!("OpenXR session exiting ({:?})", e.state()),
                        LifecycleAction::None => {}
                    }
                    self.running = self.lifecycle.running;
                }
                openxr::Event::InstanceLossPending(_) => {
                    self.lifecycle.on_state(openxr::SessionState::LOSS_PENDING);
                    self.running = false;
                }
                _ => {}
            }
        }
        // Sync the tracker action set so the action spaces have fresh poses.
        if self.running {
            let _ = self
                .session
                .sync_actions(&[openxr::ActiveActionSet::new(&self.action_set)]);
        }
    }

    /// The app has input focus (FOCUSED). Controller input is meaningless
    /// without it (system menu open, guardian, another app in front).
    pub fn focused(&self) -> bool {
        self.lifecycle.focused
    }

    /// The runtime asked the app to quit (EXITING) or lost the session
    /// (LOSS_PENDING). A standalone app should exit its loop.
    pub fn exit_requested(&self) -> bool {
        self.lifecycle.exit_requested
    }

    /// Time to locate poses at: now when the runtime converts system time,
    /// otherwise the last predicted display time.
    fn pose_time(&self) -> Option<openxr::Time> {
        if self.time_supported {
            self.instance.now().ok()
        } else {
            self.last_display_time
        }
    }

    /// Locate one action/reference space in the stage space at `time`.
    fn locate(&self, space: &openxr::Space, time: openxr::Time) -> Option<(glam::Vec3, glam::Quat)> {
        let loc = space.locate(&self.stage, time).ok()?;
        if !loc
            .location_flags
            .contains(openxr::SpaceLocationFlags::POSITION_VALID)
        {
            return None;
        }
        let p = loc.pose.position;
        let o = loc.pose.orientation;
        Some((
            glam::Vec3::new(p.x, p.y, p.z),
            glam::Quat::from_xyzw(o.x, o.y, o.z, o.w),
        ))
    }

    /// This frame's `XR_FB_face_tracking2` weights (70, in the extension's
    /// order), when the runtime tracks the face.
    pub fn face_weights(&self) -> Option<Vec<f32>> {
        let tracker = self.face_tracker?;
        let ext = self.instance.exts().fb_face_tracking2.as_ref()?;
        if !self.running {
            return None;
        }
        let time = self.pose_time()?;
        let mut weights = vec![0.0f32; 70];
        let mut confidences = vec![0.0f32; 2];
        let info = openxr::sys::FaceExpressionInfo2FB { ty: openxr::sys::FaceExpressionInfo2FB::TYPE, next: std::ptr::null(), time };
        let mut out = openxr::sys::FaceExpressionWeights2FB {
            ty: openxr::sys::FaceExpressionWeights2FB::TYPE,
            next: std::ptr::null_mut(),
            weight_count: weights.len() as u32,
            weights: weights.as_mut_ptr(),
            confidence_count: confidences.len() as u32,
            confidences: confidences.as_mut_ptr(),
            is_valid: openxr::sys::FALSE,
            is_eye_following_blendshapes_valid: openxr::sys::FALSE,
            data_source: openxr::sys::FaceTrackingDataSource2FB::VISUAL,
            time: openxr::sys::Time::from_nanos(0),
        };
        let r = unsafe { (ext.get_face_expression_weights2)(tracker, &info, &mut out) };
        (r == openxr::sys::Result::SUCCESS && out.is_valid == openxr::sys::TRUE).then_some(weights)
    }

    /// Combined eye gaze relative to the head (forward is -Z), when the runtime
    /// reports it.
    pub fn eye_gaze(&self) -> Option<glam::Quat> {
        let space = self.eye_gaze.as_ref()?;
        if !self.running {
            return None;
        }
        let time = self.pose_time()?;
        let loc = space.locate(&self.view, time).ok()?;
        if !loc.location_flags.contains(openxr::SpaceLocationFlags::ORIENTATION_VALID) {
            return None;
        }
        let o = loc.pose.orientation;
        Some(glam::Quat::from_xyzw(o.x, o.y, o.z, o.w))
    }

    /// All tracked full-body poses this frame (head + hands + waist + feet) in
    /// the stage space. Feed straight into `zestengine_core::TrackerTargets`.
    pub fn body_poses(&self) -> BodyPoses {
        if !self.running {
            return BodyPoses::default();
        }
        let Some(time) = self.pose_time() else {
            return BodyPoses::default();
        };
        BodyPoses {
            head: self.locate(&self.view, time),
            left_hand: self.locate(&self.left_hand, time),
            right_hand: self.locate(&self.right_hand, time),
            hips: self.locate(&self.hips, time),
            left_foot: self.locate(&self.left_foot, time),
            right_foot: self.locate(&self.right_foot, time),
            chest: self.locate(&self.chest, time),
            left_elbow: self.locate(&self.left_elbow, time),
            right_elbow: self.locate(&self.right_elbow, time),
            left_knee: self.locate(&self.left_knee, time),
            right_knee: self.locate(&self.right_knee, time),
        }
    }

    /// Tracked hand skeletons (left, right) in the stage space; None when a hand
    /// is not currently tracked or the runtime lacks `XR_EXT_hand_tracking`.
    pub fn hand_joints(&self) -> [Option<HandJoints>; 2] {
        if !self.running {
            return [None, None];
        }
        let Some(time) = self.pose_time() else {
            return [None, None];
        };
        std::array::from_fn(|i| {
            let tracker = self.hand_trackers[i].as_ref()?;
            let locations = self.stage.locate_hand_joints(tracker, time).ok().flatten()?;
            let mut joints = HandJoints::default();
            for (j, loc) in locations.iter().enumerate() {
                let tracked = loc.location_flags.contains(
                    openxr::sys::SpaceLocationFlags::POSITION_VALID
                        | openxr::sys::SpaceLocationFlags::POSITION_TRACKED,
                );
                joints.valid[j] = tracked;
                if tracked {
                    let p = loc.pose.position;
                    joints.positions[j] = glam::Vec3::new(p.x, p.y, p.z);
                }
                joints.radii[j] = loc.radius;
            }
            Some(joints)
        })
    }

    /// Read this frame's controller inputs (call after [`poll`], which syncs the
    /// action set). All zero/false when not running.
    pub fn input(&self) -> VrInput {
        if !self.running {
            return VrInput::default();
        }
        let f = |a: &openxr::Action<f32>| {
            a.state(&self.session, openxr::Path::NULL)
                .map(|s| s.current_state)
                .unwrap_or(0.0)
        };
        let b = |a: &openxr::Action<bool>| {
            a.state(&self.session, openxr::Path::NULL)
                .map(|s| s.current_state)
                .unwrap_or(false)
        };
        VrInput {
            left_stick: (f(&self.a_lstick_x), f(&self.a_lstick_y)),
            right_stick: (f(&self.a_rstick_x), f(&self.a_rstick_y)),
            left_grip: f(&self.a_lgrip),
            right_grip: f(&self.a_rgrip),
            left_trigger: b(&self.a_ltrigger),
            right_trigger: b(&self.a_rtrigger),
            menu: b(&self.a_menu),
            left_trigger_value: f(&self.a_ltrigger_value),
            right_trigger_value: f(&self.a_rtrigger_value),
            a: b(&self.a_a),
            b: b(&self.a_b),
            x: b(&self.a_x),
            y: b(&self.a_y),
            left_stick_click: b(&self.a_lstick_click),
            right_stick_click: b(&self.a_rstick_click),
        }
    }

    /// Create the eye swapchains. The format is the first of `preferred` (raw
    /// `VkFormat`s) the runtime offers, else an sRGB / half-float one it does.
    /// `multiview` asks for one swapchain with a layer per eye. Call once after
    /// the session is created, before [`begin_frame`].
    pub fn setup_stereo(&mut self, preferred: &[u32], multiview: bool) -> Result<StereoSetup> {
        let offered: Vec<u32> = self.session.enumerate_swapchain_formats().context("enumerating XR swapchain formats")?;
        let format = pick_swapchain_format(preferred, &offered)
            .with_context(|| format!("the OpenXR runtime offers no usable eye format (offered VkFormats {offered:?})"))?;
        let views = self
            .instance
            .enumerate_view_configuration_views(
                self.system,
                openxr::ViewConfigurationType::PRIMARY_STEREO,
            )
            .context("enumerating view configuration views")?;
        // Stereo = 2 views; both eyes use the recommended size.
        let v0 = views.first().context("no stereo views")?;
        let (w, h) = (
            v0.recommended_image_rect_width,
            v0.recommended_image_rect_height,
        );
        let eyes = views.len().max(2) as u32;
        let (count, array_size) = if multiview { (1, eyes) } else { (eyes, 1) };
        let mut swapchains = Vec::new();
        let mut images = Vec::new();
        for _ in 0..count {
            let sc = self
                .session
                .create_swapchain(&openxr::SwapchainCreateInfo {
                    create_flags: openxr::SwapchainCreateFlags::EMPTY,
                    usage_flags: openxr::SwapchainUsageFlags::COLOR_ATTACHMENT
                        | openxr::SwapchainUsageFlags::SAMPLED,
                    format,
                    sample_count: 1,
                    width: w,
                    height: h,
                    face_count: 1,
                    array_size,
                    mip_count: 1,
                })
                .context("creating XR swapchain")?;
            let imgs = sc.enumerate_images().context("enumerating XR swapchain images")?;
            swapchains.push(sc);
            images.push(imgs);
        }
        self.swapchains = swapchains;
        self.swap_images = images;
        self.view_extent = (w, h);
        self.multiview = multiview;
        Ok(StereoSetup { width: w, height: h, format, multiview })
    }

    /// Ask for the display refresh rate closest to `hz` the headset offers
    /// (`XR_FB_display_refresh_rate`). Returns the rate requested, or None when
    /// the runtime has no such control.
    pub fn request_refresh_rate(&self, hz: f32) -> Option<f32> {
        if !self.refresh_rate_supported {
            return None;
        }
        let rates = self.session.enumerate_display_refresh_rates().ok()?;
        let best = rates.iter().copied().min_by(|a, b| (a - hz).abs().total_cmp(&(b - hz).abs()))?;
        match self.session.request_display_refresh_rate(best) {
            Ok(()) => Some(best),
            Err(e) => {
                tracing::warn!("OpenXR refresh rate {best} Hz refused: {e}");
                None
            }
        }
    }

    /// Begin an XR frame: wait for the predicted display time, start the frame,
    /// and locate the per-eye views. Returns `None` when not running / should
    /// skip rendering this frame. The caller renders the scene into each eye's
    /// image (`XrEye::image`), then calls [`end_frame`].
    pub fn begin_frame(&mut self) -> Result<Option<XrFrame>> {
        if !self.running || self.swapchains.is_empty() {
            return Ok(None);
        }
        let state = self.frame_waiter.wait().context("xrWaitFrame")?;
        self.last_display_time = Some(state.predicted_display_time);
        self.frame_stream.begin().context("xrBeginFrame")?;
        if !state.should_render {
            // Still must end the frame with no layers.
            self.frame_stream
                .end(state.predicted_display_time, self.blend_mode, &[])
                .ok();
            return Ok(None);
        }
        let (_flags, views) = self
            .session
            .locate_views(
                openxr::ViewConfigurationType::PRIMARY_STEREO,
                state.predicted_display_time,
                &self.stage,
            )
            .context("xrLocateViews")?;
        let (w, h) = self.view_extent;
        let mut acquired = Vec::with_capacity(self.swapchains.len());
        for sc in &mut self.swapchains {
            let idx = sc.acquire_image().context("acquire XR image")?;
            sc.wait_image(openxr::Duration::INFINITE).context("wait XR image")?;
            acquired.push(idx);
        }
        let eye_count = if self.multiview { views.len() } else { views.len().min(self.swapchains.len()) };
        let mut eyes = Vec::with_capacity(eye_count);
        for (i, view) in views.iter().enumerate().take(eye_count) {
            let (sc, layer) = if self.multiview { (0, i as u32) } else { (i, 0) };
            let idx = acquired[sc];
            eyes.push(XrEye {
                image: self.swap_images[sc][idx as usize],
                layer,
                image_index: idx,
                view: view_to_view_matrix(view.pose),
                proj: fov_to_proj(view.fov, 0.05, 2000.0),
                pose: view.pose,
                fov: view.fov,
                width: w,
                height: h,
            });
        }
        Ok(Some(XrFrame {
            display_time: state.predicted_display_time,
            eyes,
        }))
    }

    /// Finish an XR frame: release each eye image and submit the projection
    /// composition layer to the compositor. Call after rendering all eyes.
    pub fn end_frame(&mut self, frame: XrFrame) -> Result<()> {
        for sc in &mut self.swapchains {
            sc.release_image().ok();
        }
        let (w, h) = self.view_extent;
        let rect = openxr::Rect2Di {
            offset: openxr::Offset2Di { x: 0, y: 0 },
            extent: openxr::Extent2Di {
                width: w as i32,
                height: h as i32,
            },
        };
        let proj_views: Vec<openxr::CompositionLayerProjectionView<openxr::Vulkan>> = frame
            .eyes
            .iter()
            .enumerate()
            .map(|(i, eye)| {
                let sc = if self.multiview { 0 } else { i };
                openxr::CompositionLayerProjectionView::new()
                    .pose(eye.pose)
                    .fov(eye.fov)
                    .sub_image(
                        openxr::SwapchainSubImage::new()
                            .swapchain(&self.swapchains[sc])
                            .image_array_index(eye.layer)
                            .image_rect(rect),
                    )
            })
            .collect();
        let layer = openxr::CompositionLayerProjection::new()
            .space(&self.stage)
            .views(&proj_views);
        self.frame_stream
            .end(frame.display_time, self.blend_mode, &[&layer])
            .context("xrEndFrame")?;
        Ok(())
    }

    /// The headset pose (position + orientation) in the stage reference space, or
    /// `None` if not yet tracking. Feed into `zestengine_core::TrackerTargets::head`.
    pub fn head_pose(&self) -> Option<(glam::Vec3, glam::Quat)> {
        if !self.running {
            return None;
        }
        let time = self.pose_time()?;
        let loc = self.view.locate(&self.stage, time).ok()?;
        if !loc
            .location_flags
            .contains(openxr::SpaceLocationFlags::POSITION_VALID)
        {
            return None;
        }
        let p = loc.pose.position;
        let o = loc.pose.orientation;
        Some((
            glam::Vec3::new(p.x, p.y, p.z),
            glam::Quat::from_xyzw(o.x, o.y, o.z, o.w),
        ))
    }
}

/// A face tracker for `XR_FB_face_tracking2` (visual data source), or None
/// when the runtime refuses one.
fn create_face_tracker(
    instance: &openxr::Instance,
    session: &openxr::Session<openxr::Vulkan>,
) -> Option<openxr::sys::FaceTracker2FB> {
    let ext = instance.exts().fb_face_tracking2.as_ref()?;
    let mut sources = [openxr::sys::FaceTrackingDataSource2FB::VISUAL];
    let info = openxr::sys::FaceTrackerCreateInfo2FB {
        ty: openxr::sys::FaceTrackerCreateInfo2FB::TYPE,
        next: std::ptr::null(),
        face_expression_set: openxr::sys::FaceExpressionSet2FB::DEFAULT,
        requested_data_source_count: sources.len() as u32,
        requested_data_sources: sources.as_mut_ptr(),
    };
    let mut tracker = <openxr::sys::FaceTracker2FB as openxr::sys::Handle>::NULL;
    let r = unsafe { (ext.create_face_tracker2)(session.as_raw(), &info, &mut tracker) };
    if r != openxr::sys::Result::SUCCESS {
        tracing::warn!("OpenXR face tracker not created: {r:?}");
        return None;
    }
    Some(tracker)
}

#[cfg(test)]
mod tests {
    use super::*;
    use openxr::SessionState as S;

    #[test]
    fn lifecycle_runs_a_pause_resume_cycle() {
        let mut life = SessionLifecycle::default();
        assert_eq!(life.on_state(S::IDLE), LifecycleAction::None);
        assert_eq!(life.on_state(S::READY), LifecycleAction::Begin);
        assert!(!life.running);
        life.began(true);
        assert!(life.running);
        life.on_state(S::SYNCHRONIZED);
        life.on_state(S::VISIBLE);
        assert!(life.visible && !life.focused);
        life.on_state(S::FOCUSED);
        assert!(life.visible && life.focused);
        // System menu: focus goes, frames still show.
        life.on_state(S::VISIBLE);
        assert!(life.visible && !life.focused && life.running);
        // App paused: the runtime stops the session.
        life.on_state(S::SYNCHRONIZED);
        assert_eq!(life.on_state(S::STOPPING), LifecycleAction::End);
        assert!(!life.running && !life.visible && !life.focused);
        assert_eq!(life.on_state(S::IDLE), LifecycleAction::None);
        // Resumed: READY again.
        assert_eq!(life.on_state(S::READY), LifecycleAction::Begin);
        life.began(true);
        life.on_state(S::FOCUSED);
        assert!(life.running && life.focused && !life.exit_requested);
    }

    #[test]
    fn swapchain_format_prefers_caller_then_srgb() {
        assert_eq!(pick_swapchain_format(&[50], &[43, 50, 97]), Some(50));
        assert_eq!(pick_swapchain_format(&[44], &[97, 43]), Some(43));
        assert_eq!(pick_swapchain_format(&[], &[97]), Some(97));
        assert_eq!(pick_swapchain_format(&[44], &[44]), Some(44));
        assert_eq!(pick_swapchain_format(&[], &[37]), None);
    }

    #[test]
    fn combined_view_encloses_both_eyes() {
        let fov = openxr::Fovf { angle_left: -0.9, angle_right: 0.8, angle_up: 0.85, angle_down: -0.95 };
        let mirror = openxr::Fovf { angle_left: -0.8, angle_right: 0.9, ..fov };
        let q = glam::Quat::from_rotation_y(0.4);
        let head = glam::Vec3::new(0.3, 1.6, -0.2);
        let eyes = [
            (head + q * glam::Vec3::new(-0.032, 0.0, 0.0), q, fov),
            (head + q * glam::Vec3::new(0.032, 0.0, 0.0), q, mirror),
        ];
        let (view, proj) = combined_view(&eyes, 0.05, 100.0).unwrap();
        let vp = proj * view;
        // Points along each eye's frustum edges stay inside the combined one.
        for (p, eq, f) in &eyes {
            for (tx, ty) in [
                (f.angle_left.tan(), f.angle_up.tan()),
                (f.angle_right.tan(), f.angle_down.tan()),
                (f.angle_left.tan(), f.angle_down.tan()),
                (f.angle_right.tan(), f.angle_up.tan()),
            ] {
                for depth in [0.5f32, 5.0, 50.0] {
                    let world = *p + *eq * glam::Vec3::new(tx * depth, ty * depth, -depth) * 0.999;
                    let c = vp * world.extend(1.0);
                    let n = c.truncate() / c.w;
                    assert!(c.w > 0.0 && n.x.abs() <= 1.0 && n.y.abs() <= 1.0, "{n:?} at depth {depth}");
                }
            }
        }
        let canted = [eyes[0], (eyes[1].0, glam::Quat::from_rotation_y(0.5), mirror)];
        assert!(combined_view(&canted, 0.05, 100.0).is_none());
    }

    #[test]
    fn lifecycle_exit_and_failed_begin() {
        let mut life = SessionLifecycle::default();
        life.on_state(S::READY);
        life.began(false);
        assert!(!life.running);
        assert_eq!(life.on_state(S::EXITING), LifecycleAction::Exit);
        assert!(life.exit_requested);
        let mut lost = SessionLifecycle { running: true, visible: true, focused: true, exit_requested: false };
        assert_eq!(lost.on_state(S::LOSS_PENDING), LifecycleAction::Exit);
        assert!(lost.exit_requested && !lost.running && !lost.focused);
    }
}
