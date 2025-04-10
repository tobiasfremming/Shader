#version 430 core
#define NUM_BOIDS 17
#define KINEMATICS_INDEX 20
#define NUMI 10
#define NUMF 10.0

#define MAX_ITER 5
#define TAU 6.28318530718

#define GOD_RAY_LENGTH 0.9 // higher number = shorter rays
#define GOD_RAY_FREQUENCY 32.0

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



vec3 lightPos = vec3(-2.0, -4.0, -2.0);

float fishTime;
float isJump;
float isJump2;

vec3 ccd, ccp;
int currentIndex;
vec3 repeatIndex;

// Global values for calculating color
vec3 lastFishLocalP;
vec3 lastFishRepeatIndex;
float lastFishScale;

vec3 lastDolphinLocalP;
float lastDolphinSegmentRatio; // for longitudinal variation (0 at head, 1 at tail)

// candidate
vec3 lastFishLocalPCandidate;
vec3 lastFishRepeatIndexCandidate;
float lastFishScaleCandidate;

vec3 lastDolphinLocalPCandidate;
float lastDolphinSegmentRatioCandidate; // for longitudinal variation (0 at head, 1 at tail)



float dot2( in vec2 v ) { return dot(v,v); }
float dot2( in vec3 v ) { return dot(v,v); }
float ndot( in vec2 a, in vec2 b ) { return a.x*b.x - a.y*b.y; }


float rand(vec2 co) { return fract(sin(dot(co.xy, vec2(12.9898,78.233))) * 43758.5453); }

float smoothMin(in float da, in float db, in float k){
    float h = max(k - abs(da - db), 0.0) / k;
    return min(da, db) - h * h * h * k * (1.0 / 6.0);
}

vec3 ditherNormal(vec3 normal, vec2 uv) {
    // Apply a small random offset to the normal based on the UV coordinates
    float ditherAmount = 0.02; // Adjust this value to control the dithering intensity
    vec3 ditheredNormal = normal + ditherAmount * vec3(
        rand(uv + vec2(1.0, 0.0)) - 0.5,
        rand(uv + vec2(0.0, 1.0)) - 0.5,
        rand(uv) - 0.5
    );
    return normalize(ditheredNormal);
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
    float waveFrequency = 1.0;
    float waveAmplitude = 0.1;

    float nose = 0.0;
    float tail = 0.25; // or whatever is the tail's max x in your model

    // Tail sway influence: 0.0 at nose, 1.0 at tail
    float factor = clamp((p.x - nose) / (tail - nose), 0.0, 1.0);
    factor = pow(factor, 2.0); // Exponential falloff to limit front movement

    // Apply sine wave deformation toward the Z axis
    p.z += sin(p.x * waveFrequency + (rand(repeatIndex.xy) + iTime) * 10.0) 
         * waveAmplitude * factor;

    return p;
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

// Not in use in current implementation, but nice for debugging
float sdPredator(vec3 p) {
    float scale = 2.0;
    // Scale the coordinate so the shape expands to twice its size.
    // For an SDF, the scaled SDF is: sd_scaled(p) = scale * sd_original(p/scale)
    return scale * sdFish_based(p / scale);
}

// Basic domain repeat function that wraps a coordinate into a range of [-cellSize/2, cellSize/2].
vec3 domainRepeat(vec3 p, float cellSize) {
    return mod(p + cellSize * 0.5, cellSize) - cellSize * 0.5;
}

vec3 hash3(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return fract((p.xxy + p.yzz) * p.zyx);
}


vec3 calculateFishColor3(vec3 localP, float scale) {
    // -------------------------------
    // 1. Vertical Belly-to-Back Gradient
    // -------------------------------
    // Normalize the Y coordinate relative to fish size
    float normalizedY = localP.y / scale;
    
    // Compute belly factor (1.0 at bottom, 0.0 at top)
    // Adjust these thresholds to control where the transition happens
    float bellyFactor = smoothstep(0.05, -0.1, normalizedY); // Inverted for proper blending
    
    // Define colors
    vec3 bellyColor = vec3(0.92, 0.92, 0.96);  // Silvery-white (belly)
    vec3 backColor = vec3(0.00, 0.0, 0.1);     // Dark blue (back)
    
    // Base color mix
    vec3 baseColor = mix(backColor, bellyColor, bellyFactor);
    
    // -------------------------------
    // 2. Stripe Pattern (only on back)
    // -------------------------------
    // Only apply stripes where bellyFactor is low (on the back)
    float stripeStrength = 1.0 - smoothstep(0.1, 0.6, bellyFactor);
    
    // Create stripe pattern based on horizontal position
    float stripeFreq = 100.0; // Reduced frequency for more visible stripes
    float stripePattern = sin(localP.x * stripeFreq * 2.0) * 
                         sin(localP.z * stripeFreq * 0.9);
    
    // Sharp stripe mask
    float stripeMask = step( 0.9, abs(stripePattern));
    
    // Apply stripes only to the back portion
    baseColor = mix(baseColor, vec3(0.0), stripeMask * stripeStrength);
    
    return clamp(baseColor, 0.0, 1.0);
}


vec3 calculateFishColor(vec3 localP, float scale) {
    // -------------------------------
    // 1. Vertical Belly-to-Back Gradient
    // -------------------------------
    // Compute a blend factor based on the local Y coordinate.
    // For localP.y below -0.1*scale, the fish is fully belly (silver).
    // For localP.y above 0.05*scale, it’s fully back (blue).
    float bellyFactor = smoothstep(-0.1 * scale, 0.05 * scale, localP.y);

    // Define the colors.
    vec3 bellyColor = vec3(0.92, 0.92, 0.96);  // Silvery-white (belly)
    // Adjust the back color to be darker blue
    vec3 backColor  = vec3(0.00, 0.0, 0.1);    // Dark blue (back)

    // Mix the colors with bellyFactor.
    // When bellyFactor is 0, you get bellyColor; when it's 1, you get backColor.
    vec3 baseColor = mix(bellyColor, backColor, 0.8);

    // -------------------------------
    // 2. Stripe Pattern (Tiger Stripes on the Blue part)
    // -------------------------------
    // We want stripes to appear only on the blue (back) part.
    // Thus, we define a stripe strength that activates only when bellyFactor is high.
    float stripeStrength = smoothstep(0.1, 0.5, bellyFactor);
    
    // Compute a stripe pattern based on the horizontal axis.  
    // The high frequency creates many stripes.
    float stripeFreq = 200.0;
    float stripePattern = sin(localP.x * stripeFreq + localP.z * 1.0);
    
    // Map the absolute stripe pattern to a sharp mask.
    // Using smoothstep with an inverted range gives sharp transitions.
    float stripeMask = smoothstep(0.2, 0.15, abs(stripePattern));
    
    // The stripe color is black.
    vec3 stripeColor = vec3(0.0);

    // Mix the base color with black using the stripe mask * strength.
    // In areas where stripeStrength is 0 (belly), the stripes are not applied.
    baseColor = mix(baseColor, stripeColor, stripeMask * stripeStrength);
    

    // Return the color clamped to valid RGB range
    return clamp(baseColor, 0.0, 1.0);
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
                vec3 offset = vec3(float(i), float(j), float(k));
                vec3 rid = id + offset * o;
                rid = clamp(rid, -(lim - vec3(1.0)), lim - vec3(1.0));
                repeatIndex = rid;
                // Compute a jitter offset based on the candidate cell's ID.
                // Using hash3(rid) ensures that adjacent cells vary consistently.
                vec3 cellJitter = (hash3(rid) - 0.5) * jitterAmount;
                // Compute the local coordinate within this candidate cell with jitter
                vec3 r = p - s * (rid + cellJitter);
                vec3 localP = rotationFromDirection(r, vel); // TODO: use time to add some jitter offset
                float scale = 0.5;
                float fishSDF = sdFish(localP, scale);
                
                // Calculate fish color

                if (fishSDF < d) {
                    d = fishSDF;
                    lastFishLocalPCandidate = localP;
                    lastFishRepeatIndexCandidate = rid;
                    lastFishScaleCandidate = scale;
                }
            }
        }
    }
    
    return d;
}



// DOLPHIN ===================================================================================

vec3 calculateDolphinColor(vec3 localP, float spineRatio) {
    return vec3(0.7, 0.6, 0.8); // Placeholder for dolphin color
    // Vertical gradient (Y axis): belly to top
    float bellyBlend = smoothstep(-0.1, 0.05, localP.y);

    vec3 belly = vec3(0.85, 0.85, 0.88);   // silvery gray
    vec3 back  = vec3(0.1, 0.1, 0.15);     // dark steel blue
    vec3 bodyColor = mix(belly, back, bellyBlend);

    // Optional: dorsal stripe?
    float dorsalStripe = smoothstep(0.0, 0.02, abs(localP.z));
    bodyColor = mix(bodyColor, bodyColor * 0.7, dorsalStripe);

    // Optional: darken tail slightly
    float tailDarken = smoothstep(0.6, 1.0, spineRatio);
    bodyColor *= mix(1.0, 0.75, tailDarken);

    return bodyColor;
}


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

            lastDolphinLocalPCandidate = p; // already local!
            lastDolphinSegmentRatioCandidate = ih + dis.y / NUMF; // for vertical / longitudinal blend
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


vec2 fishBound(vec3 p) {
    float d = 1e10;

    float renderIndex = 0.0;
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
        float dolphin_scale = 0.2;
        //float fishSDF = dolphin_scale * sdDolphin(localP/dolphin_scale).x;
        
        float fishSDF = dolphin_scale * sdDolphinKinematic(localP/dolphin_scale, vel,  prevVel).x;
        //float fishSDF = sdKinematicTube(p, vel, prevVel);
        
        if (fishSDF < d) {
            d = fishSDF;
            renderIndex = 1.0;
            vec3 lastFishLocalP;

            lastDolphinLocalP = lastDolphinLocalPCandidate;
            lastDolphinSegmentRatio = lastDolphinSegmentRatioCandidate;
        }
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
            
            float fishSDF = limitedDomainRepeatSDF(localP, 0.5, vec3(1.4), vel);
            //float fishSDF = sdFish(localP, 0.5);
     
            //d = min(d, fishSDF);
            if (fishSDF < d) {
                d = fishSDF;
                renderIndex = 2.0;
                lastFishLocalP = lastFishLocalPCandidate;
                lastFishRepeatIndex = lastFishRepeatIndexCandidate;
                lastFishScale = lastFishScaleCandidate;
            }
            
        } else {
            // Optionally, you can still use the sphere distance as a lower bound.
            d = min(d, sphereSDF);
        }
    }
    
    return vec2(d, renderIndex);
}

float oceanFloor(vec3 p) {
    return p.y + 1.0; // flat horizontal plane at y = -1.0
}

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}



float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);

    // Bilinear interpolation
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));

    vec2 u = f * f * (3.0 - 2.0 * f); // smoothstep

    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

float fbm(vec2 p) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;

    for (int i = 0; i < 5; i++) {
        value += amplitude * noise(p * frequency);
        frequency *= 2.0;
        amplitude *= 0.5;
    }

    return value;
}

float warpedFbm(vec2 p) {
    vec2 q = vec2(fbm(p + vec2(1.7, 9.2)),
                  fbm(p + vec2(8.3, 2.8)));

    return fbm(p + 2.0 * q);
}

vec2 terrain(vec3 p) {
    float baseHeight = p.y + 1.5;

    // Macro amplitude shaping
    float macro = fbm(p.xz * 0.15);           // slow macro undulation
    float macroHeight = mix(0.5, 1.5, macro); // varying terrain "scale"

    // Mid-frequency detail hills and ridges
    float ridges = abs(sin(p.x * 0.3) * sin(p.z * 0.3));  // simple repeating ridges
    float ridgeHeight = 0.3 * ridges;

    // Mountain height: large, blocky terrain from low-freq FBM
    float mountains = pow(fbm(p.xz * 0.07 + vec2(5.0)), 1.8); // steeper features
    float mountainHeight = 1.4 * mountains;

    // 🔀 Combine terrain components
    float finalHeight =
        macroHeight *
        (0.4 * warpedFbm(p.xz * 1.5 + iTime * 0.02) +  // small detail
         ridgeHeight +
         mountainHeight);

    return vec2(baseHeight + finalHeight, 3.0);
}

vec2 terrainBound(vec3 p) {
    if (p.y < -0.9) {
        return terrain(p);
    }
    return vec2(1000000., 0.0);
}


float coralBlob(vec3 p) {
    p.y += 1.5; // raise from floor
    float bumpy = length(p) - 0.4 + 0.1 * sin(10.0*p.x + iTime) * sin(10.0*p.y + iTime);
    return bumpy;
}
// ===========================================================================================


vec2 map(vec3 p){
    
    //return min(fishBound(p), terrain(p));
    
    //return the vec2 with lowest x value
    vec2 fishSDF = fishBound(p);
    vec2 terrainSDF = terrainBound(p);

    if (fishSDF.x < terrainSDF.x) {
        return fishSDF;
    }
    else {
        return terrainSDF;
    }

}

// LIGHTING ==========================================================

// https://iquilezles.org/articles/rmshadows/
float softShadow(vec3 ro, vec3 rd, float k) {
    float res = 1.0;
    float t = 0.02; // start distance to skip self-shadowing
    for (int i = 0; i < 24; i++) {
        float h = map(ro + rd * t).x;
        if (h < 0.001) return 0.0; // fully in shadow
        res = min(res, k * h / t);
        t += clamp(h, 0.01, 0.3); // adaptive stepping
        if (t > 10.0) break;
    }
    return clamp(res, 0.0, 1.0);
}

vec3 doLighting2(vec3 pos, vec3 normal, vec3 viewDir, float gloss, float gloss2, float shadows, vec3 baseColor, float ao) {
    vec3 lightDir = normalize(lightPos);
    vec3 halfVec = normalize(lightDir - viewDir);
    vec3 refl = reflect(-viewDir, normal);

    float sky     = clamp(normal.y, 0.0, 1.0);
    float ground  = clamp(-normal.y, 0.0, 1.0);
    float diff    = max(dot(normal, lightDir), 0.0);
    float back    = max(0.3 + 0.7 * dot(normal, -vec3(lightDir.x, 0.0, lightDir.z)), 0.0);
    float fresnel = pow(1.0 - dot(viewDir, normal), 5.0);
    float spec    = pow(max(dot(halfVec, normal), 0.0), 16.0 * gloss); // reduce gloss sharpness
    float sss     = pow(1.0 + dot(normal, viewDir), 2.0);

    // -- TUNED --
    vec3 brdf = vec3(0.0);
    brdf += 1.5 * diff * vec3(1.0)*shadows;                             // sunlight
    brdf += 0.8 * sky * vec3(0.3, 0.5, 0.7);                    // sky tint
    brdf += 0.4 * ground * vec3(0.1, 0.15, 0.1);                // soft green bounce
    brdf += 0.3 * back * vec3(0.2, 0.2, 0.25);                  // gentle backlighting
    brdf += 0.6 * sss * vec3(1.0) * gloss * ao;                 // subtle SSS
    brdf += 0.5 * spec * vec3(1.0) * fresnel * gloss * ao * shadows;      // specular
    // Optional soft reflection boost
    brdf += 0.25 * pow(max(dot(refl, lightDir), 0.0), 4.0) * gloss2;

    return baseColor * brdf;
}

// vec3 doLighting(vec3 pos, vec3 normal, vec3 viewDir, float gloss, float gloss2, float shadows, vec3 baseColor, float ao) {
//     vec3 lightDir = normalize(lightPos); // your global lightPos is fine
//     vec3 halfVec = normalize(lightDir - viewDir);
//     vec3 refl = reflect(viewDir, normal);

//     float sky     = clamp(normal.y, 0.0, 1.0);
//     float ground  = clamp(-normal.y, 0.0, 1.0);
//     float diff    = max(dot(normal, lightDir), 0.0);
//     float back    = max(0.3 + 0.7 * dot(normal, -vec3(lightDir.x, 0.0, lightDir.z)), 0.0);
//     float fresnel = pow(1.0 - dot(viewDir, normal), 5.0);
//     float spec    = pow(max(dot(halfVec, normal), 0.0), 32.0 * gloss);
//     float sss     = pow(1.0 + dot(normal, viewDir), 2.0);

//     // Optional: real shadow tracing if you have softShadow()
//     float shadowFactor = softShadow(pos + normal * 0.01, lightDir, 16.0);

//     vec3 brdf = vec3(0.0);
//     brdf += 4.0 * diff * vec3(1.00, 0.75, 0.55) * shadowFactor;
//     brdf += 1.0  * sky  * vec3(0.20, 0.45, 0.6) * (0.5 + 0.5 * ao);
//     brdf += 0.2  * back * vec3(0.40, 0.6, 0.8);
//     brdf += 1.0  * ground * vec3(0.1, 0.2, 0.15);
//     brdf += 0.8  * sss  * vec3(0.3, 0.3, 0.35) * gloss * ao;
//     brdf += 0.3  * spec * vec3(1.2, 1.1, 1.0) * shadowFactor * fresnel * gloss;

//     return baseColor * brdf;
// }


vec3 doLighting(vec3 pos, vec3 normal, vec3 viewDir, float gloss, float gloss2, float shadows, vec3 baseColor, float ao) {
    vec3 lightDir = normalize(lightPos); // your global lightPos is fine
    vec3 halfVec = normalize(lightDir - viewDir);
    vec3 refl = reflect(viewDir, normal);

    float sky     = clamp(normal.y, 0.0, 1.0);
    float ground  = clamp(-normal.y, 0.0, 1.0);
    float diff    = max(dot(normal, lightDir), 0.0);
    float back    = max(0.3 + 0.7 * dot(normal, -vec3(lightDir.x, 0.0, lightDir.z)), 0.0);
    float fresnel = pow(1.0 - dot(viewDir, normal), 5.0);
    float spec    = pow(max(dot(halfVec, normal), 0.0), 32.0 * gloss);
    float sss     = pow(1.0 + dot(normal, viewDir), 2.0);

    // Optional: real shadow tracing if you have softshadow()
    //float shadowFactor = 1.0; // TODO: replace with softshadow() if needed
    float shadowFactor = softShadow(pos + normal * 0.01, lightDir, 16.0);

    vec3 brdf = vec3(0.0);
    brdf += 20.0 * diff * vec3(1.00, 0.75, 0.55) * shadowFactor;
    brdf += 5.0  * sky  * vec3(0.20, 0.45, 0.6) * (0.5 + 0.5 * ao);
    brdf += 1.0  * back * vec3(0.40, 0.6, 0.8);
    brdf += 5.0  * ground * vec3(0.1, 0.2, 0.15);
    brdf += 4.0  * sss  * vec3(0.3, 0.3, 0.35) * gloss * ao;
    brdf += 1.5  * spec * vec3(1.2, 1.1, 1.0) * shadowFactor * fresnel * gloss;

    return baseColor * brdf;
}




vec3 getObjectColor(float renderIndex, vec3 position, vec3 normal) {
    // Lighting directions
    vec3 up = vec3(0.0, 1.0, 0.0);
    float facingUp = dot(normal, up);

    // if (renderIndex == 2.0) {
    //     // Fish - blue top, white belly, black stripes
    //     float topBlend = clamp(facingUp * 0.5 + 0.5, 0.0, 1.0);
    //     float stripe = smoothstep(0.01, 0.03, abs(sin(20.0 * position.x + position.z * 10.0)));
    //     vec3 base = mix(vec3(1.0), vec3(0.1, 0.4, 0.9), topBlend); // belly → top
    //     base = mix(base, vec3(0.0), stripe); // overlay stripes
    //     base *= 1.2; // boost color a bit
    //     //return base;
    //     return vec3(0.9, 0.2, 0.3); // for debugging

    // } else if (renderIndex == 1.0) {
    //     // Dolphin - gray top, white belly
    //     float topBlend = clamp(facingUp * 0.5 + 0.5, 0.0, 1.0);
    //     //return mix(vec3(1.0), vec3(0.9, 0.9, 0.9), topBlend); // belly → top
    //     return vec3(0.5, 0.4, 0.5); // for debugging


    // } else if (renderIndex == 3.0) {
    //     // Ocean floor - brown and green variation
    //     float noiseVal = fbm(position.xz * 0.5*position.y);
    //     vec3 dirt = vec3(1.0, 0.2, 0.08);
    //     vec3 moss = vec3(0.2, 1.0, 0.4);
    //     return mix(dirt, moss, noiseVal);
        
    // }
    float noiseVal = fbm(position.xz * 0.5*position.y);
        vec3 dirt = vec3(0.6, 0.2, 0.08)* 0.4;
        vec3 moss = vec3(0.2, 1.0, 0.4) * 0.5;
        return mix(dirt, moss, noiseVal);

    return vec3(1.0); // fallback white
}


vec3 calculate_normal(in vec3 p)
{
    const vec3 small_step = vec3(0.001, 0.0, 0.0);

    float gradient_x = map(p + small_step.xyy).x - map(p - small_step.xyy).x;
    float gradient_y = map(p + small_step.yxy).x - map(p - small_step.yxy).x;
    float gradient_z = map(p + small_step.yyx).x - map(p - small_step.yyx).x;

    vec3 normal = vec3(gradient_x, gradient_y, gradient_z);

    return normalize(normal);
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

float causticsPattern(vec3 worldPos) {
    float time = iTime * 0.5 + 23.0;
    vec2 uv = worldPos.xz * 0.5; // Change scale as needed
    vec2 p = mod(uv * TAU, TAU) - 250.0;
    vec2 i = p;

    float c = 1.0;
    float inten = 0.005;

    for (int n = 0; n < MAX_ITER; n++) {
        float t = time * (1.0 - (3.5 / float(n + 1)));
        i = p + vec2(
            cos(t - i.x) + sin(t + i.y),
            sin(t - i.y) + cos(t + i.x)
        );
        c += 1.0 / length(vec2(
            p.x / (sin(i.x + t) / inten),
            p.y / (cos(i.y + t) / inten)
        ));
    }

    c /= float(MAX_ITER);
    c = 1.17 - pow(c, 1.4);
    return clamp(pow(abs(c), 8.0), 0.0, 1.0);
}

float generateCaustics(vec3 position) {
    vec2 coord = position.xz * 0.5;
    vec2 tileCoord = mod(coord * TAU, TAU) - 250.0;
    vec2 warp = tileCoord;

    float accum = 1.0;
    float strength = 0.005;
    float phase = iTime * 0.5 + 23.0;

    for (int i = 0; i < MAX_ITER; ++i) {
        float speed = phase * (1.0 - (3.5 / float(i + 1)));

        warp = tileCoord + vec2(
            sin(warp.y + speed) + cos(warp.x - speed),
            cos(warp.x + speed) + sin(warp.y - speed)
        );

        vec2 denom = vec2(
            sin(warp.x + speed) / strength,
            cos(warp.y + speed) / strength
        );

        accum += 1.0 / length(tileCoord / denom);
    }

    accum /= float(MAX_ITER);
    float bright = 1.15 - pow(accum, 1.45);
    return clamp(pow(abs(bright), 7.5), 0.0, 1.0);
}






float drawParticle(vec2 uv, vec2 center, float radius) {
    float d = length(uv - center);
    return exp(-pow(d / radius, 2.0) * 8.0); // soft glow falloff
}

vec4 rayMarch(in vec3 ro, in vec3 rd, in vec2 uv, in vec2 uv2){
    
    int numSteps = 64;
    float threshold = 0.001;
    float distanceTraveled = 0.0;
    float radius = 1000000.0;
    const float MAXIMUM_TRACE_DISTANCE = 1000.0;
    vec3 lightPosition = lightPos;
    vec3 background = mix(vec3(0.1, 0.3, 0.5), vec3(0.788, 0.956, 1.0), uv.y*0.5);
    
    while(distanceTraveled < MAXIMUM_TRACE_DISTANCE){
    
        vec3 current_position = ro + distanceTraveled * rd;
        vec2 mapping = map(current_position);
        radius = mapping.x;
        float renderIndex = mapping.y;
        
        if (radius < threshold * distanceTraveled){
            // hit
            vec3 color = vec3(1.0, 1.0, 1.0);
            vec3 normal = calculate_normal(current_position);
            normal = ditherNormal(normal, uv); // dither normal to make it look smoother



            vec3 baseColor = vec3(1.0, 1.0, 1.0);
            if (renderIndex == 1.0) {
                baseColor = calculateDolphinColor(lastDolphinLocalP, lastDolphinSegmentRatio);
            }
            else if (renderIndex == 2.0) {
                baseColor = calculateFishColor(lastFishLocalP, lastFishScale);
            }
            else {
                baseColor = getObjectColor(renderIndex, current_position, normal);
            }
            color = baseColor;
            
            vec3 direction_to_light = normalize(current_position - lightPosition);
            
            //float diffuse_intensity = max(0.01, dot(normal, direction_to_light));
            
            float caustics = generateCaustics(current_position);
            
            caustics = pow(caustics, 0.5);   // Optional: adjust caustics intensity
            //diffuse_intensity *= 0.6 + 0.4 * caustics;  // Optional: boost contrast
            //vec3 causticsColor = vec3(0.5, 0.8, 1.0); // ocean-like tint
            //color += causticsColor * caustics * 0.6;  // blend based on strength
            color *= caustics;
            //specular
            //vec3 viewDir = normalize(current_position - ro);
            vec3 viewDir = normalize(ro - current_position);
            // vec3 reflectDir = reflect(direction_to_light, normal);
            // float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.);

            float gloss = 1.5;     // Dolphin shininess
            float gloss2 = 10.4;     // Optional reflection boost
            float ao = 1.0;         // Ambient occlusion placeholder
            float shadows = 0.0;    // (Set to 0.0 if you skip softshadow)
            vec3 lightDirection = normalize(lightPosition - current_position);
            //float shadowFactor = softShadow(current_position + normal * 0.01, lightDirection, 16.0);

            color = doLighting(current_position, normal, viewDir, gloss, gloss2, shadows, color, ao);

                        
            // // Vlumetric fog (approximate) %Report
            float boolf = 1.0; // turn effect on or off

            

            if (renderIndex == 3.0) {
                
                float godrays = GodRays(uv, uv2);
                vec3 lightColor = mix(vec3(0.5, 1.0, 0.8), vec3(0.55, 0.55, 0.95) * 0.75, 1.0 - uv.y);
                background = mix(background, lightColor, (godrays + 0.05)/1.05);
                //color *= (diffuse_intensity + spec);

                // Final fog blend based on depth
                float fogAmount = 1.0 - exp(-pow(distanceTraveled * 0.3, 1.2));
                color = mix(color, background, fogAmount);
                
            }
            else {
                //color = mix(color * (diffuse_intensity + spec), background, boolf * clamp(pow(distanceTraveled/6., 0.7), 0.0, 1.0));
                color = mix(color, background, boolf * clamp(pow(distanceTraveled/4., 0.7), 0.0, 1.0));

            }
            return vec4(color, 1.);
            
        }
        
        distanceTraveled += radius;
    
    }
    float godrays = GodRays(uv, uv2);
    vec3 lightColor = mix(vec3(0.5, 1.0, 0.8), vec3(0.55, 0.55, 0.95) * 0.75, 1.0 - uv.y);
    
    // Blend the godrays (scattered light) with the scene color.
    // Adjust the blend factor if necessary.
    background = mix(background, lightColor, (godrays + 0.05)/1.05);

    

    
    vec3 finalColor = background;
    finalColor = clamp(finalColor, 0.0, 1.0);
    // Glow pulse from small animals
    if (fract(sin( uv.x * 45.1 + iTime * 4.0 + uv.y) * 1234.56) > 0.99998) {
        finalColor += vec3(0.4, 0.7, 0.6); // aqua glow
    }

    
    //return vec4(0.4, 0.6, 0.7, 1.);
    return vec4(finalColor,1.);


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
    // TODO: add camera sway
    //vec3 ro =  target + radius * lookDir+ vec3(0.0, 0.2*sin(iTime), 0.1*cos(iTime));

    // Forward = direction from camera to target
    vec3 forward = normalize(target - ro);

    // worldUp = (0,1,0), then derive right & up for the camera
    vec3 worldUp = vec3(0.0, 1.0, 0.0);
    vec3 right   = normalize(cross(forward, worldUp));
    vec3 up      = cross(right, forward);
    
    vec3 rd = normalize(forward + uv.x * right + uv.y * up);

    vec4 result = rayMarch(ro, rd, uv, gl_FragCoord.xy/iResolution.xy);
    
    vec3 particleColor1 = vec3(0.3, 0.45, 0.35); // deeper green-brown
    vec3 particleColor2 = vec3(0.6, 0.2, 0.1);   // slightly lighter variation

    float particleOverlay = 0.0;
    vec3 particles = vec3(0.0);

    int NUM_PARTICLES = 100;
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        float id = float(i);

        // Particle base position and offset
        float speed = 0.03 + 0.015 * sin(id);
        vec2 basePos = vec2(
            fract(sin(id * 73.1) * 43758.5453 + iTime * speed),
            fract(sin(id * 91.3) * 12345.678 + iTime * speed * 0.5)
        );
        basePos.x += 0.02 * sin(iTime * 2.0 + id); // sine wave drift

        // Particle size and alpha depth blending factor
        float size = 0.004+ 0.005 * fract(sin(id * 13.37) * 123.45);
        float alpha = 0.3 + 0.7 * fract(sin(id * 45.12) * 789.23); // blend factor (for "depth")

        // Blend between two colors
        vec3 color = mix(particleColor1, particleColor2, fract(sin(id * 11.7) * 897.2));
        
        // Particle glow
        float glow = drawParticle(gl_FragCoord.xy / iResolution.xy, basePos, size)*0.5;
        particles += alpha * glow * color;
    }
    fragColor = vec4(result + vec4(particles, 0.0));
    
    //fragColor = vec4(result);
    

}


















