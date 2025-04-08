#include <chrono>
#include <GLFW/glfw3.h>
#include <glad/glad.h>
#include <SFML/Audio/SoundBuffer.hpp>
#include <utilities/shader.hpp>
#include <glm/vec3.hpp>
#include <iostream>
#include <utilities/timeutils.h>
#include <SFML/Audio/Sound.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <fmt/format.h>
#include "gamelogic.h"
#include "sceneGraph.hpp"
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/transform.hpp>

#include "utilities/imageLoader.hpp"
#include "utilities/glfont.h"

enum KeyFrameAction {
    BOTTOM, TOP
};

struct Boid {
    glm::vec3 position;
    glm::vec3 velocity;
};



constexpr int NUM_BOIDS = 10; // Number of boids in the simulation
//std::vector<Boid> boids(NUM_BOIDS);

struct GPUBoid {
    glm::vec3 position;
    glm::vec3 velocity;
};
std::vector<GPUBoid> boids(NUM_BOIDS);


#include <timestamps.h>

unsigned int currentKeyFrame = 0;
unsigned int previousKeyFrame = 0;

// TEXTURES
GLuint computeProgram;   // Compute shader program
GLuint ssbo;  // global SSBO handle

unsigned int framebuffer;
unsigned int textureColorBuffer;


// UNIFORMS
GLint u_time = -1;
GLint u_resolution = -1;
GLint u_texture = -1;

GLint posLoc = -1;
GLint velLoc = -1;

// These are heap allocated, because they should not be initialised at the start of the program
sf::SoundBuffer* buffer;
Gloom::Shader* shader;
Gloom::Shader* shader_compute;
sf::Sound* sound;

CommandLineOptions options;

bool hasStarted        = false;
bool hasLost           = false;
bool jumpedToNextFrame = false;
bool isPaused          = false;

bool mouseLeftPressed   = false;
bool mouseLeftReleased  = false;
bool mouseRightPressed  = false;
bool mouseRightReleased = false;


// Modify if you want the music to start further on in the track. Measured in seconds.
const float debug_startTime = 0;
double totalElapsedTime = debug_startTime;
double gameElapsedTime = debug_startTime;

double mouseSensitivity = 1.0;
double lastMouseX = windowWidth / 2;
double lastMouseY = windowHeight / 2;


// Constants defining boids behavior

const float ALIGNMENT_RADIUS       = 0.7f;
const float COHESION_RADIUS        = 0.8f;
const float SEPARATION_RADIUS      = 0.5f;
const float PREDATOR_FEAR_RADIUS   = 0.5f;
const float SEARCH_RADIUS          = 0.8f;

// Weights for boid behaviors
const float PREDATOR_CHASE_WEIGHT  = 1.0f;
const float ALIGNMENT_WEIGHT  = 0.4f;
const float COHESION_WEIGHT   = 0.4f;
const float SEPARATION_WEIGHT = 1.0f;
const float FEAR_WEIGHT       = 4.0f;



#include <fstream>
#include <sstream>
#include <string>

std::string loadShaderSource(const std::string& filePath) {
    std::ifstream file(filePath);
    
    if (!file.is_open()) {
        std::cerr << "Failed to open shader file: " << filePath << std::endl;
        return "";
    }

    std::stringstream buffer;
    buffer << file.rdbuf(); // Read file contents into the stringstream
    return buffer.str();    // Return as a string
}


// Function to update velocities of boids
void updateBoidVelocities(std::vector<GPUBoid>& boids) {
    std::vector<glm::vec3> newVelocities(NUM_BOIDS);

    for (int i = 0; i < NUM_BOIDS; i++) {
        glm::vec3 pos = boids[i].position;
        glm::vec3 vel = boids[i].velocity;

        if (i == 0) { // Predator (Dolphin) chases nearest prey
            float minDist = FLT_MAX;
            glm::vec3 target;

            for (int j = 2; j < NUM_BOIDS; j++) {
                float dist = glm::length(boids[j].position - pos);
                if (dist < minDist) {
                    minDist = dist;
                    target = boids[j].position;
                }
            }

            glm::vec3 chaseVec = glm::normalize(target - pos);
            glm::vec3 acceleration = chaseVec * PREDATOR_CHASE_WEIGHT;
            newVelocities[i] = glm::normalize(vel + 0.04f * acceleration);

        } else if (i == 1) { // Dolphin previous position, keep unchanged
            newVelocities[i] = vel;

        } else { // Fish behaviors
            glm::vec3 alignment(0.0f);
            glm::vec3 cohesion(0.0f);
            glm::vec3 separation(0.0f);
            glm::vec3 fear(0.0f);

            int neighborCount = 0;

            // Check fear from predator
            glm::vec3 predatorPos = boids[0].position;
            float distPredator = glm::length(predatorPos - pos);

            if (distPredator < PREDATOR_FEAR_RADIUS && distPredator > 0.001f) {
                fear = glm::normalize(pos - predatorPos) * (FEAR_WEIGHT / (distPredator * distPredator));
            }

            // Iterate through other fish
            for (int j = 2; j < NUM_BOIDS; j++) {
                if (i == j) continue;

                glm::vec3 otherPos = boids[j].position;
                glm::vec3 otherVel = boids[j].velocity;
                float dist = glm::length(otherPos - pos);

                if (dist < SEARCH_RADIUS && dist > 0.001f) {
                    if (dist < ALIGNMENT_RADIUS)
                        alignment += otherVel;
                    if (dist < COHESION_RADIUS)
                        cohesion += otherPos;
                    if (dist < SEPARATION_RADIUS)
                        separation += glm::normalize(pos - otherPos) / dist;

                    neighborCount++;
                }
            }

            if (neighborCount > 0) {
                alignment /= neighborCount;
                cohesion = (cohesion / float(neighborCount)) - pos;
                separation /= neighborCount;
            }

            glm::vec3 acceleration =
                alignment * ALIGNMENT_WEIGHT +
                cohesion * COHESION_WEIGHT +
                separation * SEPARATION_WEIGHT +
                fear;

            newVelocities[i] = glm::normalize(vel + acceleration);
        }
    }

    // Update velocities
    for (int i = 0; i < NUM_BOIDS; i++) {
        boids[i].velocity = newVelocities[i];
    }
}

// Function to update boid positions
void updateBoidPositions(std::vector<GPUBoid>& boids, float timeStep, glm::vec3 bounds) {
    for (auto& boid : boids) {
        boid.position += boid.velocity * timeStep;

        //Wrap positions
        for (int axis = 0; axis < 3; ++axis) {
            if (boid.position[axis] < -bounds[axis]) boid.position[axis] += 2.0f * bounds[axis];
            else if (boid.position[axis] > bounds[axis]) boid.position[axis] -= 2.0f * bounds[axis];
        }
    }
}












void mouseCallback(GLFWwindow* window, double x, double y) {
    int windowWidth, windowHeight;
    glfwGetWindowSize(window, &windowWidth, &windowHeight);
    glViewport(0, 0, windowWidth, windowHeight);

    double deltaX = x - lastMouseX;
    double deltaY = y - lastMouseY;

    glfwSetCursorPos(window, windowWidth / 2, windowHeight / 2);
}


SceneNode* rootNode;

void initGame(GLFWwindow* window, CommandLineOptions gameOptions) {
    buffer = new sf::SoundBuffer();
    if (!buffer->loadFromFile("../../res/Hall of the Mountain King.ogg")) {
        return;
    }

    options = gameOptions;

    for (int i = 0; i < NUM_BOIDS; ++i) {
        float x = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
        float y = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
        float z = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
        boids[i].position = glm::vec3(x * 4.f, y * 2.f, z * 2.f);

        // Random velocity
        float vx = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
        float vy = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
        float vz = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
        boids[i].velocity = glm::normalize(glm::vec3(vx, vy, vz)) * 1.2f;
    }


    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
    glfwSetCursorPosCallback(window, mouseCallback);

    shader = new Gloom::Shader();
    shader->makeBasicShader("../../res/shaders/simple.vert", "../../res/shaders/simple.frag");
    shader->activate();
    GLuint shaderProgram = shader->get();
    u_time       = glGetUniformLocation(shaderProgram, "iTime");
    u_resolution = glGetUniformLocation(shaderProgram, "iResolution");
    posLoc       = glGetUniformLocation(shaderProgram, "boidPositions");
    velLoc       = glGetUniformLocation(shaderProgram, "boidVelocities");
    u_texture    = glGetUniformLocation(shaderProgram, "iChannel0");

    // mak eshader_compute .comp shader

    GLuint computeShader = glCreateShader(GL_COMPUTE_SHADER);
    std::string computeCode = loadShaderSource("../../res/shaders/boids.comp");
    const char* codePtr = computeCode.c_str();
    glShaderSource(computeShader, 1, &codePtr, nullptr);
    glCompileShader(computeShader);

    // Check for compile errors
    GLint success;
    glGetShaderiv(computeShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(computeShader, 512, nullptr, infoLog);
        std::cerr << "ERROR: Compute shader compilation failed:\n" << infoLog << std::endl;
    }

    computeProgram = glCreateProgram();
    glAttachShader(computeProgram, computeShader);
    glLinkProgram(computeProgram);

    std::vector<GPUBoid> gpuBoids(NUM_BOIDS);


    for (int i = 0; i < NUM_BOIDS; ++i) {
        gpuBoids[i].position = glm::vec4(boids[i].position, 0.0f);
        gpuBoids[i].velocity = glm::vec4(boids[i].velocity, 0.0f);
    }

    glGenBuffers(1, &ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(GPUBoid) * NUM_BOIDS, gpuBoids.data(), GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);


    // GLuint ssbo;
    // glGenBuffers(1, &ssbo);
    // glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    // glBufferData(GL_SHADER_STORAGE_BUFFER, NUM_BOIDS * sizeof(Boid), boids.data(), GL_DYNAMIC_DRAW);
    // glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);



    // shader_compute = new Gloom::Shader();

    
    
    
    // glGenFramebuffers(1, &framebuffer);
    // glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

    
    // glGenTextures(1, &textureColorBuffer);  
    // glBindTexture(GL_TEXTURE_2D, textureColorBuffer);
    // glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, NUM_BOIDS, 2, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);   
    // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    // glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureColorBuffer, 0);




    unsigned int emptyVAO;
	glGenVertexArrays(1, &emptyVAO);
	glBindVertexArray(emptyVAO);

    getTimeDeltaSeconds();

    std::cout << "Ready. Click to start!" << std::endl;

    glm::vec2 resolution = glm::vec2(windowWidth, windowHeight); 
    //glUniform2fv(1,1, glm::value_ptr(resolution));
 
    
    rootNode = createSceneNode();
}


void renderNode(SceneNode* node) {
	switch (node->nodeType) {
	case POINT_LIGHT:
	{

	}
	break;
	}

	for (SceneNode* child : node->children) {
		renderNode(child);
	}
}

void updateFrame(GLFWwindow* window) {
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

 
    // Give the fragment shader at taste of time
    //glUniform1f(2, (float) totalElapsedTime);
    glUseProgram(shader->get()); // Just in case
    glUniform1f(u_time, static_cast<float>(totalElapsedTime));

    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_1)) {
        mouseLeftPressed = true;
        mouseLeftReleased = false;
    } else {
        mouseLeftReleased = mouseLeftPressed;
        mouseLeftPressed = false;
    }
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_2)) {
        mouseRightPressed = true;
        mouseRightReleased = false;
    } else {
        mouseRightReleased = mouseRightPressed;
        mouseRightPressed = false;
    }

    double timeDelta = getTimeDeltaSeconds();

    // TODO call funtoin to calculate the new positions of the boids
    float timeStep = 0.005f;
    glm::vec3 bounds = glm::vec3(4.f, 2.f, 2.f);
    updateBoidVelocities(boids);
    updateBoidPositions(boids, timeStep, bounds);

    std::vector<glm::vec3> positions, velocities;
    for (const auto& b : boids) {
        positions.push_back(b.position);
        velocities.push_back(b.velocity);
    }

    glUniform3fv(posLoc, NUM_BOIDS, glm::value_ptr(positions[0]));
    glUniform3fv(velLoc, NUM_BOIDS, glm::value_ptr(velocities[0]));
    //print out the posiitons and velocities of the boids
    //std::cout << "Boid positions: " << std::endl;
    // for (int i = 0; i < NUM_BOIDS; i++) {
    //     std::cout << "Boid " << i << ": " << boids[i].position.x << ", " << boids[i].position.y << ", " << boids[i].position.z << std::endl;
    // }



    // for (int i = 0; i < NUM_BOIDS; i++) {
    //     std::string posUniform = "boidPositions[" + std::to_string(i) + "]";
    //     std::string velUniform = "boidVelocities[" + std::to_string(i) + "]";

    //     GLint posLoc = glGetUniformLocation(shader->get(), posUniform.c_str());
    //     GLint velLoc = glGetUniformLocation(shader->get(), velUniform.c_str());

    //     if (posLoc != -1) {
    //         glUniform3f(posLoc, boids[i].position.x, boids[i].position.y, boids[i].position.z);
    //     } else {
    //         std::cerr << "Uniform not found: " << posUniform << std::endl;
    //     }

    //     if (velLoc != -1) {
    //         glUniform3f(velLoc, boids[i].velocity.x, boids[i].velocity.y, boids[i].velocity.z);
    //     } else {
    //         std::cerr << "Uniform not found: " << velUniform << std::endl;
    //     }
    // }

    if(!hasStarted) {
        if (mouseLeftPressed) {
            if (options.enableMusic) {
                sound = new sf::Sound();
                sound->setBuffer(*buffer);
                sf::Time startTime = sf::seconds(debug_startTime);
                sound->setPlayingOffset(startTime);
                sound->play();
            }
            totalElapsedTime = debug_startTime;
            gameElapsedTime = debug_startTime;
            hasStarted = true;
        }
        }
        else {
            totalElapsedTime += timeDelta;
            if(hasLost) {
                if (mouseLeftReleased) {
                    hasLost = false;
                    hasStarted = false;
                    currentKeyFrame = 0;
                    previousKeyFrame = 0;
                }
            } else if (isPaused) {
                if (mouseRightReleased) {
                    isPaused = false;
                    if (options.enableMusic) {
                        sound->play();
                    }
                }
            } else {
                gameElapsedTime += timeDelta;
                    if (mouseRightReleased) {
                        isPaused = true;
                        if (options.enableMusic) {
                            sound->pause();
                        }
                    }
                // Get the timing for the beat of the song
                for (unsigned int i = currentKeyFrame; i < keyFrameTimeStamps.size(); i++) {
                    if (gameElapsedTime < keyFrameTimeStamps.at(i)) {
                        continue;
                    }
                    currentKeyFrame = i;
                }

            jumpedToNextFrame = currentKeyFrame != previousKeyFrame;
            previousKeyFrame = currentKeyFrame;

            double frameStart = keyFrameTimeStamps.at(currentKeyFrame);
            double frameEnd = keyFrameTimeStamps.at(currentKeyFrame + 1); // Assumes last keyframe at infinity

            double elapsedTimeInFrame = gameElapsedTime - frameStart;
            double frameDuration = frameEnd - frameStart;
            double fractionFrameComplete = elapsedTimeInFrame / frameDuration;

            KeyFrameAction currentOrigin = keyFrameDirections.at(currentKeyFrame);
            KeyFrameAction currentDestination = keyFrameDirections.at(currentKeyFrame + 1);
        }
    }
    //glm::vec3 cameraPosition = glm::vec3(0.);
    glm::vec3 cameraPosition = glm::vec3(0, 2, -20);
    // TODO: maybe make camera move?

    //glUniform3fv(3,1, glm::value_ptr(cameraPosition));
}

void renderFrame(GLFWwindow* window) {
    int windowWidth, windowHeight;
    glfwGetWindowSize(window, &windowWidth, &windowHeight);
    glViewport(0, 0, windowWidth, windowHeight);

    glm::vec2 resolution(windowWidth, windowHeight);

    // glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    // shader_compute->activate();
    // //glUseProgram(shader_compute->get());

    // glDrawArrays(GL_POINTS, 0, NUM_BOIDS); // TODO: what should be the arguments here
    
    // glBindBuffer(GL_FRAMEBUFFER, 0);
    // glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

 

    
    glUseProgram(computeProgram);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo); // Rebind just in case
    glDispatchCompute(NUM_BOIDS, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);




    shader->activate();

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, textureColorBuffer);

    glUseProgram(shader->get()); // Make sure shader is active
    glUniform2fv(u_resolution, 1, glm::value_ptr(resolution));
    glUniform1i(u_texture, 0); // Set the texture unit to 0


    renderNode(rootNode);

    // Make the screen into two polygons forming a rectangle and draw it!
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);


}
