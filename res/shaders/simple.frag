#version 430 core
#define NUM_BOIDS 20
#define KINEMATICS_INDEX 20
#define NUMI 10
#define NUMF 10.0

#define GOD_RAY_LENGTH 1.1 // higher number = shorter rays
#define GOD_RAY_FREQUENCY 28.0

uniform float iTime;
uniform vec2 iResolution;

struct Boid {
    vec3 position;
    vec3 velocity;
};

layout(std430, binding = 0) buffer BoidBuffer {
    Boid boids[];
};



out vec4 fragColor;



vec3 lightPos = vec3(2.0, -5.0, 3.0);

float fishTime;
float isJump;
float isJump2;

vec3 ccd, ccp;
int currentIndex;
vec3 repeatIndex;

float dot2( in vec2 v ) { return dot(v,v); }
float dot2( in vec3 v ) { return dot(v,v); }
float ndot( in vec2 a, in vec2 b ) { return a.x*b.x - a.y*b.y; }


float rand(vec2 co) { return fract(sin(dot(co.xy, vec2(12.9898,78.233))) * 43758.5453); }


vec2 indexToUV(int boidIndex, vec2 textureSize) {
    vec2 pixelCoord = vec2(mod(float(boidIndex), textureSize.x),
                           floor(float(boidIndex) / textureSize.x));
    return (pixelCoord) / textureSize;
}

float smoothMin(in float da, in float db, in float k){
    float h = max(k - abs(da - db), 0.0) / k;
    return min(da, db) - h * h * h * k * (1.0 / 6.0);
}

float expSmoothMin(float a, float b, float k) {
    float res = exp(-k * a) + exp(-k * b);
    return -log(res) / k;
}

float udRoundBox( vec3 p, vec3 b, float r )
{
    return length(max(abs(p)-b,0.0))-r;
}




vec3 rotationFromDirection(vec3 p, vec3 velocity) {
    // Ensure the velocity is normalized.
    vec3 forward = -normalize(velocity); // We invert because our fish’s nose is along -X.
    
    // Choose an arbitrary up vector. If forward is nearly parallel to up, choose a different up.
    vec3 up = vec3(0.0, 1.0, 0.0);
    if (abs(dot(forward, up)) > 0.99) {
        up = vec3(1.0, 0.0, 0.0);
    }
    
    // Compute the right vector.
    vec3 right = normalize(cross(up, forward));
    // Recompute up to ensure orthonormality.
    up = cross(forward, right);
    
    // Construct a rotation matrix.
    // Here, our fish model is assumed to be modeled in local space with:
    //   - Forward (nose) along -X, Up along Y, Right along Z.
    // We want to rotate p so that model -X aligns with the world-space 'forward'.
    // One way is to form the matrix with columns corresponding to the desired world basis.
    mat3 rot = mat3(forward, up, right);
    
    // Apply the rotation to p.
    return p * rot;
}


vec3 rotate(vec3 v, float angle, vec3 axis)
{
    // Make sure the axis is normalized
    axis = normalize(axis);
    
    float c = cos(angle);
    float s = sin(angle);
    // Rodrigues’ rotation formula
    return v * c + cross(axis, v) * s + axis * dot(axis, v) * (1.0 - c);
}

// NOT WORKING CORRECTLY
vec3 rotationFromDirectionDolphin(vec3 p, vec3 velocity) {
    // First, rotate the dolphin’s model coordinates to align its nose
    // from +Z to –X (rotate -90° about the Y axis).
    p = rotate(p, 1.5708, vec3(0.0, 1.0, 0.0)); // -90° in radians

    // Then, apply the same rotation function as for the fish (which assumes nose along -X).
    return rotationFromDirection(p, velocity);
}




float sdVesicaSegment( in vec3 p, in vec3 a, in vec3 b, in float w )
{
    vec3  c = (a+b)*0.5;
    float l = length(b-a);
    vec3  v = (b-a)/l;
    float y = dot(p-c,v);
    vec2  q = vec2(length(p-c-y*v),abs(y));
    
    float r = 0.5*l;
    float d = 0.5*(r*r-w*w)/w;
    vec3  h = (r*q.x<d*(q.y-r)) ? vec3(0.0,r,0.0) : vec3(-d,0.0,d+w);
 
    return length(q-h.xy) - h.z;
}

float udTriangle( vec3 p, vec3 a, vec3 b, vec3 c )
{
  vec3 ba = b - a; vec3 pa = p - a;
  vec3 cb = c - b; vec3 pb = p - b;
  vec3 ac = a - c; vec3 pc = p - c;
  vec3 nor = cross( ba, ac );

  return sqrt(
    (sign(dot(cross(ba,nor),pa)) +
     sign(dot(cross(cb,nor),pb)) +
     sign(dot(cross(ac,nor),pc))<2.0)
     ?
     min( min(
     dot2(ba*clamp(dot(ba,pa)/dot2(ba),0.0,1.0)-pa),
     dot2(cb*clamp(dot(cb,pb)/dot2(cb),0.0,1.0)-pb) ),
     dot2(ac*clamp(dot(ac,pc)/dot2(ac),0.0,1.0)-pc) )
     :
     dot(nor,pa)*dot(nor,pa)/dot2(nor) );
}




vec3 animateFish(vec3 p) {
    float waveFrequency = 1.0;    // How many waves along the fish’s length
    float waveAmplitude = 0.1;    // Maximum vertical offset
    float fishLength = 2.5;       // Assumed x-distance from nose to tail
    float fishFactor = 0.1;

    // Create a factor that goes from 0 at the nose (x=0) to 1 at the tail (x=fishLength)
    float factor = smoothstep(0.0, fishLength*fishFactor, p.x);
    
    // Apply sine-based offset on the y coordinate, scaled by the factor.
    p.z += sin(p.x * waveFrequency + ( rand(repeatIndex.xy) + iTime)*10.) * waveAmplitude * factor;
    
    return p;
}

float sdFish2(vec3 p){
    float factor = 0.1;
    //p = animateFish(p, factor);
        //vec3 rotatedPos = rotate(p, 1.1, vec3(1.0, 0.0, 0.0));
    
    float fishLen = 2.4;
    float offset = -factor*fishLen/2.;
    float sdV = sdVesicaSegment(p, vec3(0.0 + offset , 0., 0.)*factor, vec3(1.9 + + offset, 0., 0.)*factor, 0.32*factor);
    float sdDorsalFin = udTriangle(p, vec3(0.1 + offset, 0.0, 0.0)*factor, vec3(1.1 + offset, 0.7, 0.0)*factor, vec3(1.4 + offset, 0.0, 0.0 )*factor);
    
    float sdTail = udTriangle(p, vec3(1.9 + offset, 0., 0.)*factor, vec3(2.4 + offset, 0.4, 0.0)*factor, vec3(2.4 + offset, -0.4, 0.0 )*factor);
    float sdBellyFin = udTriangle(p, vec3(0.3 + offset, 0.0, 0.0)*factor, vec3(1.0 + offset, -0.5, 0.0)*factor, vec3(1.2 + offset, 0.0, 0.0 )*factor);
    
    //float body = smoothMin(sdV,sdDorsalFin,  0.4*factor);
    //body = expSmoothMin(body, sdBellyFin, 7.*(1./factor));
    //body =  expSmoothMin(body,sdTail,  6.*(1./factor));
    float body = min(sdV,sdDorsalFin);
    body = min(body, sdBellyFin);
    body =  min(body,sdTail);
    return body;
    
}

float sdFish_based(vec3 p) {
    p = animateFish(p);

    float sdV = sdVesicaSegment(p, vec3(-0.012, 0.0, 0.0), vec3(0.178, 0.0, 0.0), 0.032);

    float sdDorsalFin = udTriangle(p, vec3(-0.002, 0.0, 0.0), vec3(0.098, 0.07, 0.0), vec3(0.128, 0.0, 0.0));

    float sdTail = udTriangle(p, vec3(0.178, 0.0, 0.0), vec3(0.228, 0.04, 0.0), vec3(0.228, -0.04, 0.0));
    
    float sdBellyFin = udTriangle(p, vec3(0.018, 0.0, 0.0), vec3(0.088, -0.05, 0.0), vec3(0.108, 0.0, 0.0));

    //float body = min(sdV, sdDorsalFin);
    //body = min(body, sdBellyFin);
    //body = min(body, sdTail);
    float body = smoothMin(sdV, sdDorsalFin, 0.04);
    body = smoothMin(body, sdBellyFin, 0.02);
    body = smoothMin(body, sdTail, 0.02);
    
    return body;
}

float sdFish(vec3 p, float scale){
    float fishSDF = scale * sdFish_based(p / scale);
    float sphereSDF = length(p) - scale * 0.3;
    // Use max to clip the fish SDF: if the fish SDF is lower than the sphere's distance,
    // it means we're inside the fish's region; otherwise, the sphere SDF takes over.
    return max(fishSDF, sphereSDF);
}

float sdPredator(vec3 p) {
    float scale = 2.0;
    // Scale the coordinate so the shape expands to twice its size.
    // For an SDF, the scaled SDF is: sd_scaled(p) = scale * sd_original(p/scale)
    return scale * sdFish_based(p / scale);
}

// This function wraps a coordinate into a range of [-cellSize/2, cellSize/2].
vec3 domainRepeat(vec3 p, float cellSize) {
    return mod(p + cellSize * 0.5, cellSize) - cellSize * 0.5;
}

vec3 hash3(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return fract((p.xxy + p.yzz) * p.zyx);
}



// Limited Domain Repetition in 3D
// p   : the point in space to evaluate
// s   : repetition period (cell size)
// lim : a vec3 specifying the maximum allowed instance ID (e.g. for a grid of 5 instances in x, lim.x should be 3.0,
//        so that the valid id range is clamped to [-2,2] after subtracting 1)
float limitedDomainRepeatSDF(vec3 p, float s, vec3 lim, vec3 vel) {
    // Compute the base cell ID from p in repeated space
    vec3 id = round(p / s);

    // Determine which side p is relative to the cell center
    vec3 o = sign(p - s * id);
    // Jitter amount relative to the cell size
    float jitterAmount = 0.34;
    
    float d = 1e20;
    
    // Loop over the 2x2x2 neighborhood
    for (int k = 0; k < 2; k++) {
        for (int j = 0; j < 2; j++) {
            for (int i = 0; i < 2; i++) {
                
                // Compute neighbor offset based on p's side relative to id
                vec3 offset = vec3(float(i), float(j), float(k));
                vec3 rid = id + offset * o;
                // Clamp the candidate cell ID to limit the repetition domain
                rid = clamp(rid, -(lim - vec3(1.0)), lim - vec3(1.0));
                repeatIndex = rid;
                // Compute a jitter offset based on the candidate cell's ID.
                // Using hash3(rid) ensures that adjacent cells vary consistently.
                vec3 cellJitter = (hash3(rid) - 0.5) * jitterAmount;
                // Compute the local coordinate within this candidate cell with jitter
                vec3 r = p - s * (rid + cellJitter);
                // Evaluate the fish SDF (and align it with the fish's orientation)
                float fishSDF = sdFish(rotationFromDirection(r, vel), 0.5);
                // Keep the minimum distance over all candidate cells
                d = min(d, fishSDF);
            }
        }
    }
    
    return d;
}
// NOT IN USE
// p: the point in space to evaluate
// s: the repetition period (cell size)
float domainRepeatSDF(vec3 p, float s) {
    // Compute the cell ID of p by rounding p/s
    vec3 id = round(p / s);
    // Compute a directional offset based on which side of the cell center p lies
    vec3 o = sign(p - s * id);
    // Initialize d to a large number
    float d = 1e20;
    
    // Loop over the current cell and its 7 neighbors (2x2x2 grid)
    for (int k = 0; k < 2; k++) {
        for (int j = 0; j < 2; j++) {
            for (int i = 0; i < 2; i++) {
                float scale = (k + j + i )* 0.1;
                // Compute the candidate cell ID. The expression (vec3(i, j, k)*o) selects the
                // neighboring cell along each dimension in the proper direction.
                vec3 rid = id + vec3(float(i), float(j), float(k)) * o;
                // Compute the local coordinate within that cell
                vec3 r = p - s * rid;
                // Evaluate the base SDF at the repeated coordinate and take the minimum distance
                d = min(d, sdFish(r, scale));
            }
        }
    }
    return d;
}



float confinedFishSDF(vec3 localP, float fishSDF, float boundRadius) {
    float sphereSDF = length(localP) - boundRadius;
    // Use max to clip the fish SDF: if the fish SDF is lower than the sphere's distance,
    // it means we're inside the fish's region; otherwise, the sphere SDF takes over.
    return max(fishSDF, sphereSDF);
}


// DOLPHIN ===================================================================================

float sdEllipsoid( in vec3 p, in vec3 r ) 
{
    float k0 = length(p/r);
    float k1 = length(p/(r*r));
    return k0*(k0-1.0)/k1;
}

// https://iquilezles.org/articles/distfunctions/
vec2 sd2Segment(vec3 start, vec3 end, vec3 point) {
    vec3 startToPoint = point - start;
    vec3 segmentVector = end - start;
    // Project the query point onto the segment vector and clamp it between 0 and 1
    float t = clamp(dot(startToPoint, segmentVector) / dot(segmentVector, segmentVector), 0.0, 1.0);

    vec3 closest = start + t * segmentVector;
    vec3 diff = point - closest;
    // Return squared distance to the segment and the projection factor (t)
    return vec2(dot(diff, diff), t);
}

vec2 anima( float ih, float t )
{
    float an1 = 0.9*(0.5+0.2*ih)*cos(5.0*ih - 3.0*t + 6.2831/4.0);
    float an2 = 1.0*cos(3.5*ih - 1.0*t + 6.2831/4.0);
    float an = mix( an1, an2, isJump );
    float ro = 0.4*cos(4.0*ih - 1.0*t)*(1.0-0.5*isJump);
	return vec2( an, ro );
}

vec3 anima2( void )
{
    vec3 a1 = vec3(0.0,        sin(3.0*fishTime+6.2831/4.0),0.0);
    vec3 a2 = vec3(0.0,1.5+2.5*cos(1.0*fishTime),0.0);
	vec3 a = mix( a1, a2, isJump );
	a.y *= 0.5;
	a.x += 0.1*sin(0.1 - 1.0*fishTime)*(1.0-isJump);
    return a;
}

// Helper: build a rotation matrix that rotates around 'axis' by 'angle'
mat3 rotationMatrix(vec3 axis, float angle) {
    float s = sin(angle);
    float c = cos(angle);
    float oc = 1.0 - c;
    return mat3(
        oc*axis.x*axis.x + c,         oc*axis.x*axis.y - axis.z*s,  oc*axis.z*axis.x + axis.y*s,
        oc*axis.x*axis.y + axis.z*s,    oc*axis.y*axis.y + c,         oc*axis.y*axis.z - axis.x*s,
        oc*axis.z*axis.x - axis.y*s,    oc*axis.y*axis.z + axis.x*s,  oc*axis.z*axis.z + c
    );
}



vec2 sdDolphin2(vec3 p, vec3 vel, vec3 prevVel) {
    // Initialize our result vector.
    vec2 res = vec2(1000.0, 0.0);
    
    // Transform p into local space with the head at the origin.
    
    
    // -------------------------------------------------------------------
    // Compute head rotation: rotate from the default +x direction to vel.
    // This rotation will be used only for the head SDF.
    // -------------------------------------------------------------------
    vec3 from = prevVel;
    vec3 to = normalize(vel);
    float cosAngle = dot(from, to);
    float angle = acos(cosAngle);
    vec3 axis = normalize(cross(from, to));
    // Fallback if the vectors are nearly parallel.
    //if(length(axis) < 0.001) axis = vec3(0.0, 1.0, 0.0);
    mat3 headRot = rotationMatrix(axis, angle);
    
    // -------------------------------------------------------------------
    // Build the kinematic chain for the dolphin’s body.
    // The chain is built from the head (at 0,0,0) extending along +x.
    // Only the head will be rotated; the chain follows in unrotated space.
    // -------------------------------------------------------------------
    vec3 segmentStart = vec3(0.0);
    //vec3 currentDir = vec3(1.0, 0.0, 0.0); // base direction along +x
    vec3 currentDir = axis;
    
    // Variables to store segments for fins and tail.
    vec3 p1 = segmentStart; vec3 d1 = vec3(0.0);
    vec3 p2 = segmentStart; vec3 d2 = vec3(0.0);
    vec3 p3 = segmentStart; vec3 d3 = vec3(0.0);
    vec3 midpoint = segmentStart;
    
    for (int i = 0; i < NUMI; i++) {
        float ih = float(i) / NUMF;
        vec2 anim = anima(ih, fishTime);
        float ll = 0.48;
        if (i == 0) ll = 0.655;
        // The target direction is primarily +x with a small oscillatory offset.
        vec3 targetDir = normalize(vec3(1.0, 0.2 * sin(anim.x), 0.2 * sin(anim.y)));

        float kinematicFactor = 0.01; // lower values mean more lag
        //currentDir = normalize(mix(currentDir, targetDir, kinematicFactor));
        float lagFactor = kinematicFactor * (1.0 + 0.5 * float(i)); // increases with segment index
        currentDir = normalize(mix(currentDir, targetDir, lagFactor));
        vec3 segmentEnd = segmentStart + ll * currentDir;
        
        // Compute the SDF from p to this segment.
        vec2 dis = sd2Segment(segmentStart, segmentEnd, p);
        if (dis.x < res.x) {
            res = vec2(dis.x, ih + dis.y / NUMF);
            midpoint = mix(segmentStart, segmentEnd, dis.y);
        }
        
        // Save specific segments for fin and tail SDFs.
        if (i == 3) { 
            p1 = segmentStart; 
            d1 = segmentEnd - segmentStart;
        }
        if (i == 4) { 
            p3 = segmentStart; 
            d3 = segmentEnd - segmentStart;
        }
        if (i == (NUMI - 1)) { 
            p2 = segmentEnd; 
            d2 = segmentEnd - segmentStart;
        }
        
        segmentStart = segmentEnd;
    }
    ccp = midpoint;
    
    // -------------------------------------------------------------------
    // Body SDF: use the computed midpoint from the kinematic chain.
    // -------------------------------------------------------------------
    float h = res.y;
    float ra = 0.05 + h * (1.0 - h) * (1.0 - h) * 2.7;
    ra += 7.0 * max(0.0, h - 0.04) * exp(-30.0 * max(0.0, h - 0.04))
          * smoothstep(-0.1, 0.1, p.y - midpoint.y);
    ra -= 0.03 * smoothstep(0.0, 0.1, abs(p.y - midpoint.y)) * (1.0 - smoothstep(0.0, 0.1, h));
    ra += 0.05 * clamp(1.0 - 3.0 * h, 0.0, 1.0);
    ra += 0.035 * (1.0 - smoothstep(0.0, 0.025, abs(h - 0.1)))
          * (1.0 - smoothstep(0.0, 0.1, abs(p.y - midpoint.y)));
    
    float bodySDF = 0.75 * (distance(p, midpoint) - ra);
    
    // -------------------------------------------------------------------
    // Head SDF: only compute this for the head region (e.g. p.x < 0.15).
    // Here we rotate p into head space using the headRot.
    // -------------------------------------------------------------------
    float headSDF = 1000.0;
    if (p.x < 0.15) {
        // Since headRot is orthonormal, its inverse is its transpose.
        vec3 pHead = transpose(headRot) * p;
        headSDF = sdEllipsoid(pHead, vec3(0.15, 0.1, 0.1));
    }
    
    // Blend head and body SDFs in the head region.
    float combinedSDF = bodySDF;
    if (p.x < 0.15)
        combinedSDF = smoothMin(headSDF, bodySDF, 0.1);
    
    // -------------------------------------------------------------------
    // FIN and TAIL: Use the saved segments to add fins and tail details.
    // (Same as your original logic.)
    // -------------------------------------------------------------------
    
    // Fin from segment at i==4 (using p3, d3)
    d3 = normalize(d3);
    float kVal = sqrt(1.0 - d3.y * d3.y);
    mat3 ms = mat3( d3.z / kVal, -d3.x * d3.y / kVal, d3.x,
                    0.0,         kVal,              d3.y,
                   -d3.x / kVal, -d3.y * d3.z / kVal, d3.z );
    vec3 ps = p - p3;
    ps = ms * ps;
    ps.z -= 0.4;
    float dFin = length(ps.yz) - 0.9;
    dFin = max(dFin, -(length(ps.yz - vec2(0.6, 0.0)) - 0.35));
    dFin = max(dFin, udRoundBox(ps + vec3(0.0, -0.5, 0.5), vec3(0.0, 0.5, 0.5), 0.02));
    combinedSDF = smoothMin(combinedSDF, dFin, 0.1);
    
    // Fin from segment at i==3 (using p1, d1)
    d1 = normalize(d1);
    kVal = sqrt(1.0 - d1.y * d1.y);
    ms = mat3( d1.z / kVal, -d1.x * d1.y / kVal, d1.x,
               0.0,         kVal,              d1.y,
              -d1.x / kVal, -d1.y * d1.z / kVal, d1.z );
    ps = p - p1;
    ps = ms * ps;
    ps.x = abs(ps.x);
    float lVal = ps.x;
    lVal = clamp((lVal - 0.4) / 0.5, 0.0, 1.0);
    lVal = 4.0 * lVal * (1.0 - lVal);
    lVal *= 1.0 - clamp(5.0 * abs(ps.z + 0.2), 0.0, 1.0);
    ps += vec3(-0.2, 0.36, -0.2);
    dFin = length(ps.xz) - 0.8;
    dFin = max(dFin, -(length(ps.xz - vec2(0.2, 0.4)) - 0.8));
    dFin = max(dFin, udRoundBox(ps + vec3(0.0, 0.0, 0.0), vec3(1.0, 0.0, 1.0), 0.015 + 0.05 * lVal));
    combinedSDF = smoothMin(combinedSDF, dFin, 0.12);
    
    // Tail from the final segment (using p2, d2)
    d2 = normalize(d2);
    mat2 mf = mat2(d2.z, d2.y, -d2.y, d2.z);
    vec3 pf = p - p2 - d2 * 0.25;
    pf.yz = mf * pf.yz;
    float dTail = length(pf.xz) - 0.6;
    dTail = max(dTail, -(length(pf.xz - vec2(0.0, 0.8)) - 0.9));
    dTail = max(dTail, udRoundBox(pf, vec3(1.0, 0.005, 1.0), 0.005));
    combinedSDF = smoothMin(combinedSDF, dTail, 0.1);
    
    return vec2(combinedSDF, res.y);
}


vec3 lerp(vec3 a, vec3 b,float t){

    return a * (1.0 - t) + b * t; // basically mix

}



vec2 sdDolphinKinematic(vec3 p, vec3 vel, vec3 prevVel){
    float lagFactor = 0.4; // adjust this to increase or decrease the lag
    vec3 dirPrev = -normalize(prevVel);
    vec3 dirCurr = -normalize(vel);
    vec3 blendedDir = normalize(mix(dirPrev, dirCurr, lagFactor));
    if (dot(dirPrev, dirCurr) < 0.01){
        blendedDir = dirCurr;
    }
    
    
    vec2 res = vec2( 1000.0, 0.0 );

	vec3 segmentStart = anima2();
	
	float or = 0.0;
	float th = 0.0;
	float hm = 0.0;

	vec3 p1 = segmentStart; vec3 d1=vec3(0.0);
	vec3 p2 = segmentStart; vec3 d2=vec3(0.0);
	vec3 p3 = segmentStart; vec3 d3=vec3(0.0);
	vec3 midpoint = segmentStart;
	for( int i=0; i<NUMI; i++ )
	{	
		float ih = float(i)/NUMF;
		vec2 anim = anima( ih, fishTime );
		float ll = 0.48; 
        if( i==0 ) ll=0.655;
		//vec3 segmentEnd = segmentStart + ll*normalize(vec3(sin(anim.y), sin(anim.x), cos(anim.x)));
        
        // Use the animation to create an oscillation offset.
        vec3 animOffset = normalize(vec3(sin(anim.y), sin(anim.x), cos(anim.x)));

        // Now blend the animation offset with the lagged base direction.
        vec3 laggedSegmentDir = normalize(mix(blendedDir, animOffset, lagFactor));

        // Compute the segment endpoint.
        vec3 segmentEnd = segmentStart + ll * laggedSegmentDir;

		
		vec2 dis = sd2Segment( segmentStart, segmentEnd, p );

		if( dis.x<res.x ) {
            res=vec2(dis.x,ih+dis.y/NUMF); 
            midpoint=segmentStart+(segmentEnd-segmentStart)*dis.y; 
            ccd = segmentEnd-segmentStart;
        }
		
		if( i==3 ) { 
            p1 = segmentStart; 
            d1 = segmentEnd-segmentStart;
        }
		if( i==4 ) { 
            p3=segmentStart; 
            d3 = segmentEnd-segmentStart; 
        }
        
		if( i==(NUMI-1) ) { 
            p2 = segmentEnd; 
            d2 = segmentEnd-segmentStart; 
        }

		segmentStart = segmentEnd;
	}
	ccp = midpoint;
	
	float h = res.y;
	float ra = 0.05 + h*(1.0-h)*(1.0-h)*2.7;
	ra += 7.0*max(0.0,h-0.04)*exp(-30.0*max(0.0,h-0.04)) * smoothstep(-0.1, 0.1, p.y-midpoint.y);
	ra -= 0.03*(smoothstep(0.0, 0.1, abs(p.y-midpoint.y)))*(1.0-smoothstep(0.0,0.1,h));
	ra += 0.05*clamp(1.0-3.0*h,0.0,1.0);
    ra += 0.035*(1.0-smoothstep( 0.0, 0.025, abs(h-0.1) ))* (1.0-smoothstep(0.0, 0.1, abs(p.y-midpoint.y)));
	
	// body
	res.x = 0.75 * (distance(p,midpoint) - ra);

    // fin	
	//d3 = normalize(d3);
    
    vec3 d1_current = normalize(d1);
    vec3 d2_current = normalize(d2);
    vec3 d3_current = normalize(d3);

    vec3 laggedD1 = normalize(mix(blendedDir, d1_current, lagFactor));
    vec3 laggedD2 = normalize(mix(blendedDir, d2_current, 1.));
    vec3 laggedD3 = normalize(mix(blendedDir, d3_current, lagFactor));
    
	float k = sqrt(1.0 - laggedD3.y*laggedD3.y);
	mat3 ms = mat3(  laggedD3.z/k, -laggedD3.x*laggedD3.y/k, laggedD3.x,
				        0.0,            k, laggedD3.y,
				    -laggedD3.x/k, -laggedD3.y*laggedD3.z/k, laggedD3.z );
    
	vec3 ps = p - p3;
	ps = ms*ps;
	ps.z -= 0.4;
    float d5 = length(ps.yz) - 0.9;
	d5 = max( d5, -(length(ps.yz-vec2(0.6,0.0)) - 0.35) );
	d5 = max( d5, udRoundBox( ps+vec3(0.0,-0.5,0.5), vec3(0.0,0.5,0.5), 0.02 ) );
	res.x = smoothMin( res.x, d5, 0.1 );
	
    // fin	
	// FIN from segment at i==3 (using p1, d1)

    vec3 finDir2 = normalize(laggedD1);

    // Choose an up reference and adjust if necessary.
    vec3 upRefFin2 = vec3(0.0, 1.0, 0.0);
    if (abs(dot(finDir2, upRefFin2)) > 0.99) {
        upRefFin2 = vec3(1.0, 0.0, 0.0);
    }

    // Compute the right vector and corrected up vector.
    vec3 rightFin2 = normalize(cross(upRefFin2, finDir2));
    vec3 newUpFin2 = cross(finDir2, rightFin2);

    // Build the 3×3 rotation matrix.
    mat3 finRot2 = mat3(rightFin2, newUpFin2, finDir2);

    // Transform the point into fin local space.
    vec3 finLocal2 = transpose(finRot2) * (p - p1);

    // In this fin’s SDF your original code takes abs of the x-component:
    finLocal2.x = abs(finLocal2.x);

    // Now compute a factor for additional shape adjustments.
    float lVal = finLocal2.x;
    lVal = clamp((lVal - 0.4) / 0.5, 0.0, 1.0);
    lVal = 4.0 * lVal * (1.0 - lVal);
    lVal *= 1.0 - clamp(5.0 * abs(finLocal2.z + 0.2), 0.0, 1.0);

    // Apply an offset if desired.
    finLocal2 += vec3(-0.2, 0.36, -0.2);

    // Compute the SDF for this fin.
    float dFin2 = length(finLocal2.xz) - 0.8;
    dFin2 = max(dFin2, -(length(finLocal2.xz - vec2(0.2, 0.4)) - 0.8));
    dFin2 = max(dFin2, udRoundBox(finLocal2, vec3(1.0, 0.0, 1.0), 0.015 + 0.05 * lVal));
    res.x = smoothMin(res.x, dFin2, 0.12);

	
    // Tail transformation using a full 3x3 rotation matrix

    // Compute the instantaneous tail direction d2 and blend if needed (already done):
   
    // (Assume laggedD2 is computed as you want; here we reuse it)
    vec3 tailDir = normalize(laggedD2);

    // Choose an up reference, avoiding near-parallel cases.
    vec3 upRef = vec3(0.0, 1.0, 0.0);
    if (abs(dot(tailDir, upRef)) > 0.99) {
        upRef = vec3(1.0, 0.0, 0.0);
    }

    // Compute right and newUp to build the tail coordinate system.
    vec3 right = normalize(cross(upRef, tailDir));
    vec3 newUp = cross(tailDir, right);

    // Construct the rotation matrix.
    mat3 tailRot = mat3(right, newUp, tailDir);

    // Transform the point into tail local space.
    // Here, tailDir * 0.25 is an offset along the tail direction; adjust as needed.
    vec3 tailLocal = transpose(tailRot) * (p - p2 - tailDir * 0.25);

    // Now compute the SDF in tail-local space. For instance, using tailLocal.xz:
    float dTail = length(tailLocal.xz) - 0.6;
    dTail = max(dTail, -(length(tailLocal.xz - vec2(0.0, 0.8)) - 0.9));
    dTail = max(dTail, udRoundBox(tailLocal, vec3(1.0, 0.005, 1.0), 0.005));
    res.x = smoothMin(res.x, dTail, 0.1);
	
	return res;


}



vec2 sdDolphin( vec3 p ){
    vec2 res = vec2( 1000.0, 0.0 );

	vec3 segmentStart = anima2();
	
	float or = 0.0;
	float th = 0.0;
	float hm = 0.0;

	vec3 p1 = segmentStart; vec3 d1=vec3(0.0);
	vec3 p2 = segmentStart; vec3 d2=vec3(0.0);
	vec3 p3 = segmentStart; vec3 d3=vec3(0.0);
	vec3 midpoint = segmentStart;
	for( int i=0; i<NUMI; i++ )
	{	
		float ih = float(i)/NUMF;
		vec2 anim = anima( ih, fishTime );
		float ll = 0.48; 
        if( i==0 ) ll=0.655;
		vec3 segmentEnd = segmentStart + ll*normalize(vec3(sin(anim.y), sin(anim.x), cos(anim.x)));
		
		vec2 dis = sd2Segment( segmentStart, segmentEnd, p );

		if( dis.x<res.x ) {
            res=vec2(dis.x,ih+dis.y/NUMF); 
            midpoint=segmentStart+(segmentEnd-segmentStart)*dis.y; 
            ccd = segmentEnd-segmentStart;
        }
		
		if( i==3 ) { 
            p1 = segmentStart; 
            d1 = segmentEnd-segmentStart;
        }
		if( i==4 ) { 
            p3=segmentStart; 
            d3 = segmentEnd-segmentStart; 
        }
        
		if( i==(NUMI-1) ) { 
            p2 = segmentEnd; 
            d2 = segmentEnd-segmentStart; 
        }

		segmentStart = segmentEnd;
	}
	ccp = midpoint;
	
	float h = res.y;
	float ra = 0.05 + h*(1.0-h)*(1.0-h)*2.7;
	ra += 7.0*max(0.0,h-0.04)*exp(-30.0*max(0.0,h-0.04)) * smoothstep(-0.1, 0.1, p.y-midpoint.y);
	ra -= 0.03*(smoothstep(0.0, 0.1, abs(p.y-midpoint.y)))*(1.0-smoothstep(0.0,0.1,h));
	ra += 0.05*clamp(1.0-3.0*h,0.0,1.0);
    ra += 0.035*(1.0-smoothstep( 0.0, 0.025, abs(h-0.1) ))* (1.0-smoothstep(0.0, 0.1, abs(p.y-midpoint.y)));
	
	// body
	res.x = 0.75 * (distance(p,midpoint) - ra);

    // fin	
	d3 = normalize(d3);
	float k = sqrt(1.0 - d3.y*d3.y);
	mat3 ms = mat3(  d3.z/k, -d3.x*d3.y/k, d3.x,
				        0.0,            k, d3.y,
				    -d3.x/k, -d3.y*d3.z/k, d3.z );
	vec3 ps = p - p3;
	ps = ms*ps;
	ps.z -= 0.4;
    float d5 = length(ps.yz) - 0.9;
	d5 = max( d5, -(length(ps.yz-vec2(0.6,0.0)) - 0.35) );
	d5 = max( d5, udRoundBox( ps+vec3(0.0,-0.5,0.5), vec3(0.0,0.5,0.5), 0.02 ) );
	res.x = smoothMin( res.x, d5, 0.1 );
	
    // fin	
	d1 = normalize(d1);
	k = sqrt(1.0 - d1.y*d1.y);
	ms = mat3(  d1.z/k, -d1.x*d1.y/k, d1.x,
				   0.0,            k, d1.y,
               -d1.x/k, -d1.y*d1.z/k, d1.z );
	ps = p - p1;
	ps = ms*ps;
	ps.x = abs(ps.x);
	float l = ps.x;
	l=clamp( (l-0.4)/0.5, 0.0, 1.0 );
	l=4.0*l*(1.0-l);
	l *= 1.0-clamp(5.0*abs(ps.z+0.2),0.0,1.0);
	ps.xyz += vec3(-0.2,0.36,-0.2);
    d5 = length(ps.xz) - 0.8;
	d5 = max( d5, -(length(ps.xz-vec2(0.2,0.4)) - 0.8) );
	d5 = max( d5, udRoundBox( ps+vec3(0.0,0.0,0.0), vec3(1.0,0.0,1.0), 0.015+0.05*l ) );
	res.x = smoothMin( res.x, d5, 0.12 );
	
    // tail	
	d2 = normalize(d2);
	mat2 mf = mat2( d2.z, d2.y, -d2.y, d2.z );
	vec3 pf = p - p2 - d2*0.25;
	pf.yz = mf*pf.yz;
    float d4 = length(pf.xz) - 0.6;
	d4 = max( d4, -(length(pf.xz-vec2(0.0,0.8)) - 0.9) );
	d4 = max( d4, udRoundBox( pf, vec3(1.0,0.005,1.0), 0.005 ) );
	res.x = smoothMin( res.x, d4, 0.1 );
	
	return res;
}



float fishBound(vec3 p) {
    float d = 1e10;
    // Adjust this radius so it tightly bounds your fish.
    float boundingRadius = 2.4; // make this smaller
        
    vec3 pos = boids[0].position;
    // Compute SDF for a sphere centered at pos.
    float sphereSDF = length(p - pos) - boundingRadius;
    //if (sphereSDF > 0.0) { continue;}

    if (sphereSDF < 0.01) {
        // p is inside the sphere, so we now do the more expensive evaluation.
        vec3 vel = boids[0].velocity;
        vec3 prevVel = boids[1].velocity;
        vec3 localP = p - pos;  // translate into boid's space
        //localP = rotationFromDirectionDolphin(localP, vel);
        float dolphin_scale = 0.2;
        //float fishSDF = dolphin_scale * sdDolphin(localP/dolphin_scale).x;
        
        float fishSDF = dolphin_scale * sdDolphinKinematic(localP/dolphin_scale, vel,  prevVel).x;
        //float fishSDF = sdKinematicTube(p, vel, prevVel);
        
        d = min(d, fishSDF);
    } 

    boundingRadius = 0.5;
    for (int i = 2; i < NUM_BOIDS; i++) { // start from 2?
        currentIndex = i;
        vec3 pos = boids[i].position;
        float sphereSDF = length(p - pos) - boundingRadius;
        
        if (sphereSDF < 0.01) {
            vec3 vel = boids[i].velocity;
            vec3 localP = p - pos;  // translate into boid's space
            rotationFromDirection(localP, vel);
            
            float fishSDF = limitedDomainRepeatSDF(localP, 0.5, vec3(1.5), vel);
            //float fishSDF = sdFish(localP, 0.5);
     
            d = min(d, fishSDF);
            
        } else {
            // Optionally, you can still use the sphere distance as a lower bound.
            d = min(d, sphereSDF);
        }
    }
    
    // Experimenting with previous dolphin position and velocity for kinematics
    // ===============================
    //boidUV = indexToUV(NUM_BOIDS, vec2(NUM_BOIDS, 1.0));
    //    pos = texture(iChannel0, boidUV).xyz;
    //    sphereSDF = length(p - pos) - boundingRadius;
    //    
    //    if (sphereSDF < 0.01) {
    //        vec3 vel = texture(iChannel1, boidUV).xyz;
    //        vec3 localP = p - pos;  // translate into boid's space
   
            
           
    //        localP = rotationFromDirectionDolphin(localP, vel);
    
    //        float dolphin_scale = 0.1;
    //        float fishSDF = dolphin_scale * sdDolphin(localP/dolphin_scale).x;

    //        d = min(d, fishSDF);
            
    //    }
    
    // ===============================
    
    
    return d;
}





// ===========================================================================================




float map(vec3 p){
    
    
    return fishBound(p);
   


    
}


// LIGHTING ==========================================================

vec3 calculate_normal(in vec3 p)
{
    const vec3 small_step = vec3(0.001, 0.0, 0.0);

    float gradient_x = map(p + small_step.xyy) - map(p - small_step.xyy);
    float gradient_y = map(p + small_step.yxy) - map(p - small_step.yxy);
    float gradient_z = map(p + small_step.yyx) - map(p - small_step.yyx);

    vec3 normal = vec3(gradient_x, gradient_y, gradient_z);

    return normalize(normal);
}

vec2 getLightScreenPos(vec3 lightPos, vec3 ro, vec3 forward, vec3 right, vec3 up) {
    // Compute the vector from camera to light
    vec3 lightVec = lightPos - ro;
    // Get the depth (distance along forward)
    float z = dot(lightVec, forward);
    // Project the light’s x and y onto the camera plane
    float x = dot(lightVec, right);
    float y = dot(lightVec, up);
    // Perspective divide (assuming a simple pinhole projection)
    vec2 proj = vec2(x, y) / z;
    // This result is now in normalized device coordinates (roughly -1 to 1).
    // (Adjust if your uv space is defined differently.)
    return proj;
}

float GodRays(  in vec2 ndc, in vec2 uv) {
    vec2 godRayOrigin = ndc + vec2(-1.15, -1.25);
    float rayInputFunc = atan(godRayOrigin.y, godRayOrigin.x) * 0.63661977236; // that's 2/pi
    float light = (sin(rayInputFunc * GOD_RAY_FREQUENCY + iTime * -2.25) * 0.5 + 0.5);
    light = 0.5 * (light + (sin(rayInputFunc * 13.0 + iTime) * 0.5 + 0.5));
    //light *= (sin(rayUVFunc * 8.0 + -iTime * 0.25) * 0.5 + 0.5);
    light *= pow(clamp(dot(normalize(-godRayOrigin), normalize(ndc - godRayOrigin)), 0.0, 1.0), 2.5);
    light *= pow(uv.y, GOD_RAY_LENGTH);
    light = pow(light, 1.75);
    return light;
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
            color = mix(color * (diffuse_intensity + spec), background, boolf * pow(distanceTraveled/5., 1.));
            return vec4(color, 1.);
            
        }
        
        distanceTraveled += radius;
    
    }
    float godrays = GodRays(uv, uv2);
    vec3 lightColor = mix(vec3(0.5, 1.0, 0.8), vec3(0.55, 0.55, 0.95) * 0.75, 1.0 - uv.y);
    
    // Blend the godrays (scattered light) with the scene color.
    // Adjust the blend factor if necessary.
    background = mix(background, lightColor, (godrays + 0.05)/1.05);
    
    //return vec4(0.4, 0.6, 0.7, 1.);
    return vec4(background,1.);


}




void main()
{

    vec2 uv = (2.0 * gl_FragCoord.xy - iResolution.xy) / iResolution.y;
    fishTime = 0.6 + 2.0*iTime;
    
    // =======================================================
    // mouse
    // =======================================================

    vec3 iMouse = vec3(0.0, 0.0, 0.0);
    vec2 mouse = (iMouse.z > 0.0) ? iMouse.xy : 0.5 * iResolution.xy; 
    
    // convert to yaw and pitch
    float sensitivity = 0.03;
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

    vec4 result = rayMarch(ro, rd, uv, gl_FragCoord.xy/iResolution.xy);
    
    fragColor = vec4(result);
    
    
    
 
    
}


















