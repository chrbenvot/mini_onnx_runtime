#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include <opencv2/opencv.hpp>
#include <iostream>

// --- Helper: Manage OpenGL Texture ---
// OpenCV gives us CPU data (cv::Mat). ImGui needs a GPU Texture ID (GLuint).
struct GLTexture {
    GLuint id = 0;
    int width = 0;
    int height = 0;

    void update(const cv::Mat& mat) {
        if (mat.empty()) return;

        // 1. Create Texture ID if we don't have one
        if (id == 0) {
            glGenTextures(1, &id);
            glBindTexture(GL_TEXTURE_2D, id);
            // Setup filtering parameters (LINEAR makes it smooth)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); 
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); 
        } else {
            glBindTexture(GL_TEXTURE_2D, id);
        }

        // 2. Upload Pixels
        // OpenCV is BGR, OpenGL prefers RGB. We'll fix this in the loop for efficiency, 
        // assuming 'mat' is already RGB here.
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, mat.cols, mat.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, mat.data);
        
        width = mat.cols;
        height = mat.rows;
    }

    ~GLTexture() {
        if (id != 0) glDeleteTextures(1, &id);
    }
};

int main(int, char**) {
    // 1. Setup Window (GLFW)
    if (!glfwInit()) return 1;
    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

    GLFWwindow* window = glfwCreateWindow(1280, 720, "YOLOv2 Dashboard", NULL, NULL);
    if (window == NULL) return 1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable V-Sync

    // 2. Setup ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // 3. Setup OpenCV Webcam
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open webcam." << std::endl;
        return -1;
    }

    GLTexture webcam_tex;
    cv::Mat frame, rgb_frame;

    // --- Main Loop ---
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // A. Capture Frame
        cap >> frame;
        if (!frame.empty()) {
            // Convert BGR (OpenCV) -> RGB (ImGui/OpenGL)
            cv::cvtColor(frame, rgb_frame, cv::COLOR_BGR2RGB);
            // Upload to GPU
            webcam_tex.update(rgb_frame);
        }

        // B. Start ImGui Frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // C. Define the GUI Layout
        {
            ImGui::Begin("Camera Feed");
            
            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
            
            if (webcam_tex.id != 0) {
                // Draw the image
                // (void*)(intptr_t) casts the GLuint to the type ImGui expects
                ImGui::Image((void*)(intptr_t)webcam_tex.id, ImVec2((float)webcam_tex.width, (float)webcam_tex.height));
            } else {
                ImGui::Text("Waiting for camera...");
            }

            ImGui::End();
        }

        // D. Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.45f, 0.55f, 0.60f, 1.00f);
        glClear(GL_COLOR_BUFFER_BIT);
        
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
