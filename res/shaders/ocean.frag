
// Created by Tobias Fremming
// This work is under the MIT License. Using this code or working upon it is therefore allowed.
// If you use this script, I would appriciate a shoutout.

#version 430 core
#define PI 3.14159265359
#define MAX_ITER 5
#define TAU 6.28318530718
#define fragCoord gl_FragCoord.xy
#define iMouse vec3(0.0) // mouse position

uniform float iTime;
uniform vec2 iResolution;



out vec4 fragColor;


float hash( in vec2 p ) {
    p  = fract(p * vec2(123.34, 345.45));
    p += dot(p, p + 34.345);
    return fract(p.x * p.y);
}

float noise( in vec2 p ){
    vec2 i = floor(p);
    vec2 f = fract(p);

    // smooth interpolation
    vec2 u = f * f * (3.0 - 2.0*f);

    // mix 4 corners
    float a = hash(i + vec2(0.0,0.0));
    float b = hash(i + vec2(1.0,0.0));
    float c = hash(i + vec2(0.0,1.0));
    float d = hash(i + vec2(1.0,1.0));

    return mix( mix(a,b,u.x),
                mix(c,d,u.x), u.y );
}


float cheapFbm(in vec2 p){
    float v = 0.0;
    float amp = 0.5;
    float freq = 1.0;
    // only 3 octaves instead of 6
    for(int i=0; i<3; i++){
        v   += amp * noise(p * freq);
        freq *= 2.0;
        amp  *= 0.5;
    }
    return v;
}


vec3 gerstnerWave(vec2 pos, vec2 dir, float amplitude, float wavelength, float speed) {
    float k     = 2.0 * PI / wavelength;
    float phase = speed * iTime;
    float f     = k * dot(dir, pos) + phase;
    float cosF  = cos(f);
    float sinF  = sin(f);

    // horizontal displacement (Q*A / k) * D
    // Q is often a “steepness” factor – here baked into amplitude
    vec2 horiz = dir * (amplitude * cosF / k);

    float vert = amplitude * sinF;
    return vec3(horiz.x, vert, horiz.y);
}


vec3 gerstnerWaveSharp(
    vec2 pos,
    vec2 dir,
    float amplitude,
    float wavelength,
    float speed,
    float sharpness
) {
    float k     = TAU / wavelength;
    float phase = speed * iTime;
    float f     = k * dot(dir, pos) + phase;

    float s = sin(f);
    float c = cos(f);

    // shape the waveform: sign(s)*|s|^sharpness
    float shape = sign(s) * pow(abs(s), sharpness);

    // vertical displacement is now pinched
    float vert = amplitude * shape;

    // scaled by sharpness so sharper peaks pull harder
    float drag = sharpness * amplitude * c / k;
    vec2 horiz = dir * drag;

    return vec3(horiz.x, vert, horiz.y);
}






float ocean(vec3 p){

      vec2 direction = vec2(0.5, 0.9);
      float time = sin(iTime) * 0.2 + 23.0;
      vec2 uvW = p.xz; 
      float height = 0.0;
      height += gerstnerWave(uvW, direction, 0.15, 3.5, 0.5).y; // only vertical
      
      height += gerstnerWave(uvW, normalize(vec2( 0.8, -0.6)), 0.02*0.2, 0.6,  2.5).y;
      height += gerstnerWave(uvW, normalize(vec2( 0.3,  0.7)), 0.01*0.3,  0.3, 4.0).y;
      const vec2 dir1 = normalize(vec2( 1.0,  0.2));
      const vec2 dir2 = normalize(vec2(-0.7,  1.0));
      const vec2 dir3 = normalize(vec2( 0.3, -0.8));
      float shoreFactor= 0.5;
      float shoalAmp = mix(0.01, 0.5, shoreFactor);  

      vec2 shoreNormal = vec2(0.0, 1.0); // if shoreline runs along X, normal points +Z
      vec2 refractedDir = normalize( mix(dir1, shoreNormal, shoreFactor*0.5) );

      height += gerstnerWaveSharp(uvW, refractedDir, 0.03*shoalAmp, 8.0, 1.0, 1.2).y*2.0;
      
      height += gerstnerWaveSharp(uvW, dir1, 0.03,  8.0, 1.0, 1.8).y;
      height += gerstnerWaveSharp(uvW, direction, 0.06,  3.5, 0.5, 1.5).y;
      height += gerstnerWaveSharp(uvW, dir3, 0.01, 1.0, 3.0, 0.9).y;

      float micro = cheapFbm(uvW * 8.0 + iTime * 1.2) * 0.02;
        
      height += micro;

      float detail = 0.0;
      float detail2 = 0.0;
      if (length(p) < 9.0){
          
          height += gerstnerWave(uvW, vec2(0.5, 0.6), 0.05, 1., 1.2).y; // only vertical
      
      }
      if (length(p) < 25.0){
          detail = cheapFbm(uvW * time*0.05);
          detail -= cheapFbm(uvW * time*0.5)*0.01;
          //detail += detail = fbm(uvW * time*0.03); // expensive
          
      }
      
      if (length(p) < 7.0){
          detail2 = cheapFbm(uvW * time*0.5);
      }
      
      float detail_factor = 0.25;
      float detail_factor2 = 0.02;
      return p.y - height + 1.0 + detail * detail_factor + detail2 * detail_factor2;

}

float map(vec3 p){

    return ocean(p);
    
    //if (p.y < 0.1){ // bounding optional    
    //    return ocean(p);
    //}
    //return 10000.0;
}


vec3 calculate_normal(in vec3 p)
{
    const vec3 small_step = vec3(0.001, 0.0, 0.0);

    float gradient_x = map(p + small_step.xyy) - map(p - small_step.xyy);
    float gradient_y = map(p + small_step.yxy) - map(p - small_step.yxy);
    float gradient_z = map(p + small_step.yyx) - map(p - small_step.yyx);

    vec3 normal = vec3(gradient_x, gradient_y, gradient_z);

    return normalize(normal);
}


vec3 fresnelSchlick(float cosTheta, vec3 F0) {

    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}


vec3 mixCloudsWithGradient(vec3 skyColor, vec3 cloudColor, float cloudMask, vec2 uv) {

    float t = clamp(uv.y, 0.0, 1.0);
    t = smoothstep(0.0, 1.0, t);
    float brightness = mix(0.7, 0.9,  t);
    vec3 shadedCloud = cloudColor * brightness;

    return mix(skyColor, shadedCloud, cloudMask);
}



vec4 rayMarch(in vec3 ro, in vec3 rd, in vec2 uv, in vec2 uv2){
    
    int numSteps = 64;
    float threshold = 0.001;
    float distanceTraveled = 0.0;
    float radius = 1000000.0;
    const float MAXIMUM_TRACE_DISTANCE = 1000.0;
    vec3 lightPosition = vec3(0.0, -5.0, 6.0);
    vec3 background = mix(vec3(0.788, 0.956, 1.0), vec3(0.1, 0.3, 0.7), uv.y*0.5);
    vec3 current_position = vec3(0.0);
    
    while(distanceTraveled < MAXIMUM_TRACE_DISTANCE){
    
        current_position = ro + distanceTraveled * rd;
        radius = map(current_position);
        
        if (radius < threshold * distanceTraveled){
            // hit
            vec3 color = vec3(0.00, 0.12, 0.25);
            vec3 normal = calculate_normal(current_position);
            
            // 1) compute view & light
            vec3 viewDir  = normalize(ro - current_position);
            vec3 lightDir = normalize(-lightPosition + current_position);

            // 2) lambert + spec
            float diff = max(dot(normal, lightDir), 0.0);
            vec3 H = normalize(viewDir + lightDir);
            float spec = pow(max(dot(normal, H), 0.0), 32.0);

            // 3) Fresnel term
            float cosVN = max(dot(viewDir, normal), 0.0);
            vec3 F0    = vec3(0.0);                // water reflectance at normal
            vec3 F     = fresnelSchlick(cosVN, F0);
            vec3 halfway = normalize(viewDir + lightDir);
            float  NdotH = max(dot(normal, halfway), 0.0);
            float  shininess = 256.0;
            float  glint = pow(NdotH, shininess) * 5.0;  

            // 4) composite
            vec3 baseColor  = vec3(0.00, 0.12, 0.25) * diff + + glint;  // diffuse tint
            vec3 reflectCol = mix(vec3(0.788, 0.956, 1.0), vec3(0.1, 0.3, 0.7), uv.y*0.5); 
            
            vec3 colorOut   = mix(baseColor, reflectCol, F);

            // 5) add specular “sparkle”
            if (length(current_position) < 20.0) colorOut += spec * F;    // modulate spec by Fresnel too
            

            // 6) fog
            const float fogDensity = 0.03;
            float  fog        = 1. - exp(-distanceTraveled * fogDensity);
            colorOut = mix(colorOut, background, fog);
            
            return vec4(colorOut, 1.0);

            
        }
        
        distanceTraveled += radius;
    
    }
 
    vec2 skyUV = rd.xy * vec2(0.5, 0.3) + vec2(0.0, iTime*0.01);  
    
    float cloudMask = smoothstep(0.6, 0.75, cheapFbm(skyUV*1.1 + iTime*0.01 + 23.));

    vec3 finalSky = mixCloudsWithGradient(background, vec3(0.85), cloudMask, skyUV);

    finalSky = mix(finalSky, background, 1.-skyUV.y);
    
    
    float horizonFadeRange = 0.3; 
    float horizonFade      = smoothstep(0.0, horizonFadeRange, uv.y);
    finalSky = mix(background, finalSky, horizonFade);
    
    vec3 sunColor = vec3(1.0, 0.95, 0.8) * 2.5; // todo: SUN

    return vec4(finalSky,1.);


}




void main()
{

    vec2 uv = (2.0 * fragCoord - iResolution.xy) / iResolution.y;
 
    
    // =======================================================
    // mouse
    // =======================================================

    vec2 mouse = (iMouse.z > 0.0) ? iMouse.xy : 0.5 * iResolution.xy; 
    // convert to yaw and pitch
    float sensitivity = 0.005;
    vec2 mouseDelta = mouse - 0.5 * iResolution.xy;
    float yaw   =  sensitivity * mouseDelta.x;
    float pitch = -sensitivity * mouseDelta.y;
    pitch = clamp(pitch, -1.57, 1.57);
    
    
    
    // =======================================================
    // Orbit camera
    // =======================================================
    float radius = 2.0;             // how far from origin
    vec3 target = vec3(0.0, 0.0, 0.0);
    
    // Camera direction from angles (spherical coords)
    vec3 lookDir = normalize(vec3(
        cos(pitch)*sin(yaw),
        sin(pitch),
        cos(pitch)*cos(yaw)
    ));

    // Camera position
    vec3 ro = target + radius * lookDir;

    // Forward = direction from camera to target
    vec3 forward = normalize(target - ro);

    // worldUp = (0,1,0), then derive right & up for the camera
    vec3 worldUp = vec3(0.0, 1.0, 0.0);
    vec3 right   = normalize(cross(forward, worldUp));
    vec3 up      = cross(right, forward);
    

    
    vec3 rd = normalize(forward + uv.x * right + uv.y * up);

    
    
    vec4 result = rayMarch(ro, rd, uv, fragCoord/iResolution.xy);

    fragColor = vec4(result);
    
    
    
 
    
}