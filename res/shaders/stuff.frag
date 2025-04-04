#version 430 core
#define NUM_BOIDS 5

uniform float iTime;
uniform vec2 iResolution;
uniform vec3 boidPositions[NUM_BOIDS];
uniform vec3 boidVelocities[NUM_BOIDS];

out vec4 fragColor;

float rand(vec2 co) { return fract(sin(dot(co.xy, vec2(12.9898,78.233))) * 43758.5453); }
float dither(vec2 uv) { return (rand(uv)*2.0-1.0) / (256.0); }

vec4 line(vec2 uv, float v, float h, vec3 col){
    h += boidVelocities[0].x;
    float wave = sin(iTime * v + uv.x * h*1.7)*0.87;
    uv.y += wave * smoothstep(1.0, 0.0, abs(uv.x));
    float thickness = smoothstep(0.1, 0.0, abs(uv.y));
    float fade = smoothstep(1.0, 0.2, abs(uv.y)) * smoothstep(1.0, 0.3, abs(uv.x) * boidPositions[0].x);
    return vec4(col * thickness * fade, 1.0);

}

void main()
{
    vec2 uv = gl_FragCoord.xy / iResolution.xy;

    // Scale so that x is in the range [0,1]
    uv.x = uv.x * 2.0 - 1.0;  // Centered between -1 and 1 for consistency
    uv.y = (2.0 * gl_FragCoord.y - iResolution.y) / iResolution.y;  // Keep y centered
    float num_waves = 20.;
    
    for (float i = 0.0; i <= num_waves; i += 1.0) {
        float t = i / num_waves*2.;
        //fragColor += line(uv, 1., t, vec3(1.0 - t, abs(1.-t)*0.3, t * 0.6));
        fragColor += line(uv, 1.+t*0.001, t, vec3(1.0 - t*1.5, abs(1.-t)*0.3, t * 0.6))+ vec4(dither(uv));
    }
}











