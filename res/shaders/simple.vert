#version 430 core

out vec2 uv;

const vec2 pos[4] = vec2[](
    vec2(-1.0, -1.0),
    vec2( 1.0, -1.0),
    vec2(-1.0,  1.0),
    vec2( 1.0,  1.0)
);

void main() {
    gl_Position = vec4(pos[gl_VertexID], 0.0, 1.0);
    uv = (pos[gl_VertexID] + 1.0) * 0.5; // for debug if you need
}
