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

constexpr int NUM_BOIDS = 5;
std::vector<Boid> boids(NUM_BOIDS);

#include <timestamps.h>

unsigned int currentKeyFrame = 0;
unsigned int previousKeyFrame = 0;

GLint u_time = -1;
GLint u_resolution = -1;

GLint posLoc = -1;
GLint velLoc = -1;

// These are heap allocated, because they should not be initialised at the start of the program
sf::SoundBuffer* buffer;
Gloom::Shader* shader;
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

bool still = true;

// Modify if you want the music to start further on in the track. Measured in seconds.
const float debug_startTime = 0;
double totalElapsedTime = debug_startTime;
double gameElapsedTime = debug_startTime;

double mouseSensitivity = 1.0;
double lastMouseX = windowWidth / 2;
double lastMouseY = windowHeight / 2;

void mouseCallback(GLFWwindow* window, double x, double y) {
    int windowWidth, windowHeight;
    glfwGetWindowSize(window, &windowWidth, &windowHeight);
    glViewport(0, 0, windowWidth, windowHeight);

    double deltaX = x - lastMouseX;
    double deltaY = y - lastMouseY;

    glfwSetCursorPos(window, windowWidth / 2, windowHeight / 2);
}



void updateBoids(float dt) {
    const float SEARCH_RADIUS = 0.8f;
    const float SEPARATION_RADIUS = 0.5f;
    const float PREDATOR_FEAR_RADIUS = 0.5f;
    const float PREDATOR_CHASE_WEIGHT = 1.0f;

    std::vector<Boid> newBoids = boids;

    for (int i = 0; i < NUM_BOIDS; ++i) {
        glm::vec3 pos = boids[i].position;
        glm::vec3 vel = boids[i].velocity;

        if (i == 0) {
            // Predator logic
            float minDist = 1e10;
            glm::vec3 target;
            for (int j = 1; j < NUM_BOIDS; ++j) {
                float d = glm::length(boids[j].position - pos);
                if (d < minDist) {
                    minDist = d;
                    target = boids[j].position;
                }
            }
            glm::vec3 acceleration = glm::normalize(target - pos) * PREDATOR_CHASE_WEIGHT;
            vel = glm::normalize(vel + 0.04f * acceleration);
        } else {
            // Prey logic
            glm::vec3 sumVel(0.0f), sumPos(0.0f), separation(0.0f), fear(0.0f);
            int count = 0;

            glm::vec3 predatorPos = boids[0].position;
            float distPred = glm::length(predatorPos - pos);
            if (distPred < PREDATOR_FEAR_RADIUS && distPred > 0.001f) {
                fear = glm::normalize(pos - predatorPos) * (4.0f * PREDATOR_FEAR_RADIUS) / (distPred * distPred);
            }

            for (int j = 1; j < NUM_BOIDS; ++j) {
                if (j == i) continue;
                float d = glm::length(boids[j].position - pos);
                if (d > 0.001f && d < SEARCH_RADIUS) {
                    sumVel += boids[j].velocity;
                    sumPos += boids[j].position;
                    float factor = 1.0f - glm::smoothstep(0.0f, SEPARATION_RADIUS, d);
                    separation += (pos - boids[j].position) * factor / d;
                    count++;
                }
            }

            if (count > 0) {
                glm::vec3 alignment = sumVel / float(count);
                glm::vec3 cohesion = (sumPos / float(count)) - pos;
                separation /= float(count);
                glm::vec3 acceleration = alignment * 0.4f + cohesion * 0.3f + separation * 0.8f + fear;

                vel = glm::normalize(vel + acceleration) * (2.5f + glm::length(fear));
            }
        }

        // Update position
        glm::vec3 newPos = pos + vel * dt;

        // Wrap position
        glm::vec3 bounds = glm::vec3(4.0f, 2.0f, 2.0f);
        for (int k = 0; k < 3; ++k) {
            if (newPos[k] < -bounds[k]) newPos[k] += 2.0f * bounds[k];
            else if (newPos[k] > bounds[k]) newPos[k] -= 2.0f * bounds[k];
        }

        newBoids[i].position = newPos;
        newBoids[i].velocity = vel;
    }

    boids = newBoids;
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

    updateBoids(static_cast<float>(timeDelta));


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
    glUseProgram(shader->get()); // Make sure shader is active
    glUniform2fv(u_resolution, 1, glm::value_ptr(resolution));

    std::vector<glm::vec3> positions, velocities;
    for (const auto& b : boids) {
        positions.push_back(b.position);
        velocities.push_back(b.velocity);
    }

    glUniform3fv(posLoc, NUM_BOIDS, glm::value_ptr(positions[0]));
    glUniform3fv(velLoc, NUM_BOIDS, glm::value_ptr(velocities[0]));
    renderNode(rootNode);

    // Make the screen into two polygons forming a rectangle and draw it!
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);


}
