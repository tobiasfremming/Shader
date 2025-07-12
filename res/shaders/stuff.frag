// NOT WORKING HERE YET; JUST WORK IN PROGRESS


#define numOctaves 3

float hash1( vec2 p )
{
    p  = 50.0*fract( p*0.3183099 );
    return fract( p.x*p.y*(p.x+p.y) );
}

float hash1( float n )
{
    return fract( n*17.0*fract( n*0.3183099 ) );
}

vec2 hash2( vec2 p ) 
{
    const vec2 k = vec2( 0.3183099, 0.3678794 );
    float n = 111.0*p.x + 113.0*p.y;
    return fract(n*fract(k*n));
}

float hash2d(vec2 p) {
  return fract(sin(dot(p,vec2(12.9898,78.233)))*43758.5453);
}

float noise( in vec3 x )
{
    vec3 p = floor(x);
    vec3 w = fract(x);
    
    #if 1
    vec3 u = w*w*w*(w*(w*6.0-15.0)+10.0);
    #else
    vec3 u = w*w*(3.0-2.0*w);
    #endif
    


    float n = p.x + 317.0*p.y + 157.0*p.z;
    
    float a = hash1(n+0.0);
    float b = hash1(n+1.0);
    float c = hash1(n+317.0);
    float d = hash1(n+318.0);
    float e = hash1(n+157.0);
	float f = hash1(n+158.0);
    float g = hash1(n+474.0);
    float h = hash1(n+475.0);

    float k0 =   a;
    float k1 =   b - a;
    float k2 =   c - a;
    float k3 =   e - a;
    float k4 =   a - b - c + d;
    float k5 =   a - c - e + g;
    float k6 =   a - b - e + f;
    float k7 = - a + b + c - d + e - f - g + h;

    return -1.0+2.0*(k0 + k1*u.x + k2*u.y + k3*u.z + k4*u.x*u.y + k5*u.y*u.z + k6*u.z*u.x + k7*u.x*u.y*u.z);
}

// 2D noise
float noise2d(vec2 p){
  vec2 i = floor(p), f = fract(p);
  vec2 u = f*f*(3.0-2.0*f);
  float a = hash2d(i+vec2(0,0));
  float b = hash2d(i+vec2(1,0));
  float c = hash2d(i+vec2(0,1));
  float d = hash2d(i+vec2(1,1));
  return mix(mix(a,b,u.x), mix(c,d,u.x), u.y);
}

// cheap 2D FBM (3 octaves)
float fbm2d(vec2 p) {
  float v = 0.0, amp = 0.5;
  for(int i=0;i<3;i++){
    v   += amp * noise2d(p);
    p   *= 2.0;
    amp *= 0.5;
  }
  return v;
}

float fbm( in vec3 p, in float H )
{    
    float G = exp2(-H);
    float f = 1.0;
    float a = 1.0;
    float t = 0.0;
    for( int i=0; i<numOctaves; i++ )
    {
        t += a*noise(f*p);
        f *= 2.0;
        a *= G;
    }
    return t;
}

float terrain(in vec3 p){
    return  p.y + 0.5 + fbm2d(p.xz)*0.9 + noise2d(p.xz)*0.1 - fbm2d(vec2(p.x, p.y + 0.5))*0.9;

}



float sdCylinder(vec3 p, float R){
    // distance from p.xz to circle of radius R
    return length(p.xz) - R;
}

float sdCapsule(vec3 p, vec2 a_b){
    // capsule between points a_b.x and a_b.y on the y-axis
    // here a_b = vec2(-L/2, +L/2)
    p.y = clamp(p.y, a_b.x, a_b.y);
    float R = 0.5;
    return length(p.xz) - R;
}


// bend around the X-axis: p.z gets “pulled” as a function of p.y
vec3 bendX(vec3 p, float strength){
    float c = cos(strength * p.y);
    float s = sin(strength * p.y);
    mat2 m = mat2(c, -s,
                  s,  c);
    // rotate the (z,y) plane → (newZ,newY)
    vec2 zy = m * p.zy;
    p.z = zy.x;
    p.y = zy.y;
    return p;
}


// spine parameters
const float L1 = 1.5;      // length of main body
const float L2 = 1.0;      // length of tail segment
const float R  = 0.2;      // cylinder radius


// evaluate your whale-body SDF
float sdWhaleBody(vec3 p){
    
    
    // time-varying bend strengths
    float bend1 = sin(iTime*0.8) * 0.2; 
    float bend2 = sin(iTime*1.2 + 1.0) * 0.3;
    // move p so that the spine’s origin is at (0,0,0)
    // and the y-axis points along the local “forward” of the whale

    // First segment: head+body
    vec3 p1 = bendX(p, bend1);
    float d1 = sdCapsule(vec3(p1.x, p1.y, p1.z), vec2(0.0, L1)) - R;

    // Second segment: tail, start at y=L1
    // translate p into the tail’s local frame
    vec3 p_tail = p - vec3(0.0, L1, 0.0);
    p_tail = bendX(p_tail, bend2);
    float d2 = sdCapsule(vec3(p_tail.x, p_tail.y, p_tail.z), vec2(0.0, L2)) - R;

    // union of the two segments
    return min(d1, d2);
}





float map(in vec3 p){
    return min(terrain(p), sdWhaleBody(p));
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



vec4 rayMarch(in vec3 ro, in vec3 rd, in vec2 uv, in vec2 uv2){
    
    int numSteps = 64;
    float threshold = 0.001;
    float distanceTraveled = 0.0;
    float radius = 1000000.0;
    const float MAXIMUM_TRACE_DISTANCE = 1000.0;
    vec3 lightPosition = vec3(2.0, -5.0, 3.0);
    vec3 background = mix(vec3(0.1, 0.3, 0.5), vec3(0.788, 0.956, 1.0), uv.y*0.5);
    
    while(distanceTraveled < MAXIMUM_TRACE_DISTANCE){
    
        vec3 current_position = ro + distanceTraveled * rd;
        radius = map(current_position);
        
        if (radius < threshold * distanceTraveled){
            // hit
            vec3 color = vec3(1.0, 1.0, 1.0);
            vec3 normal = calculate_normal(current_position);
            
            vec3 direction_to_light = normalize(current_position - lightPosition);
            
            float diffuse_intensity = max(0.0, dot(normal, direction_to_light));
            
            // specular
            vec3 viewDir = normalize(current_position - ro);
            vec3 reflectDir = reflect(direction_to_light, normal);
            float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.);
            
            float boolf = 1.0; // turn effect on or off
            //color = mix(color * (diffuse_intensity + spec), background, boolf * pow(distanceTraveled/5., 1.));
            //return vec4(color, 1.);
            
            color = color * (diffuse_intensity + spec);
            
            const float fogDensity = 0.3;
            float  fog        = 1. - exp(-distanceTraveled * fogDensity);
            vec3 colorOut = mix(color, background, fog);
            
            return vec4(colorOut, 1.0);
            
        }
        
        distanceTraveled += radius;
    
    }
    vec3 lightColor = mix(vec3(0.5, 1.0, 0.8), vec3(0.55, 0.55, 0.95) * 0.75, 1.0 - uv.y);
   
    
    //return vec4(0.4, 0.6, 0.7, 1.);
    return vec4(background,1.);


}





void mainImage( out vec4 fragColor, in vec2 fragCoord )
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
    vec3 target = vec3(-sin(iTime*0.2) *4.3, 0.0, -iTime);
    
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