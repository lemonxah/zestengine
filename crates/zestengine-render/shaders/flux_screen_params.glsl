// Copyright (c) 2026 Ryno Kotze.
// ZestEngine License v1.1 (source-available); see the repository LICENSE and NOTICE.
// Third-party rights and licenses, where applicable, are not replaced.

struct FluxScreenParams {
    mat4 view;
    mat4 projection;
    mat4 inverse_projection;
    mat4 previous_view_projection;
    mat4 previous_inverse_view_projection;
    vec4 previous_camera;
    vec4 options;
};
