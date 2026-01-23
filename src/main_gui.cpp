#include "engine.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm> // For std::max, std::min

// --- Helper: Manage OpenGL Texture ---
struct GLTexture
{
    GLuint id = 0;
    int width = 0;
    int height = 0;

    void update(const cv::Mat &mat)
    {
        if (mat.empty())
            return;
        if (id == 0)
        {
            glGenTextures(1, &id);
            glBindTexture(GL_TEXTURE_2D, id);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        }
        else
        {
            glBindTexture(GL_TEXTURE_2D, id);
        }
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, mat.cols, mat.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, mat.data);
        width = mat.cols;
        height = mat.rows;
    }

    ~GLTexture()
    {
        if (id != 0)
            glDeleteTextures(1, &id);
    }
};

// --- Helper: Preprocessing ---
std::vector<float> preprocess(const cv::Mat &src)
{
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(416, 416));
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

    std::vector<float> output;
    output.reserve(3 * 416 * 416);

    std::vector<cv::Mat> channels(3);
    cv::split(resized, channels);

    for (int c = 0; c < 3; ++c)
    {
        for (int y = 0; y < 416; ++y)
        {
            for (int x = 0; x < 416; ++x)
            {
                output.push_back(static_cast<float>(channels[c].at<uint8_t>(y, x))); // 0.0 to 255.0
            }
        }
    }
    return output;
}

// --- YOLO CONSTANTS ---
const int GRID_W = 13;
const int GRID_H = 13;
const int NUM_ANCHORS = 5;
const int NUM_CLASSES = 20;
const int BLOCK_SIZE = 5 + NUM_CLASSES;

const float ANCHORS[10] = {1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52};

const char *CLASS_NAMES[] = {
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"};

struct Detection
{
    float x, y, w, h;
    float confidence;
    int class_id;
    std::string label;
    int timer = 0;
};

// --- MATH HELPERS ---
inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

float iou(const Detection &a, const Detection &b)
{
    float x1 = std::max(a.x - a.w / 2, b.x - b.w / 2);
    float y1 = std::max(a.y - a.h / 2, b.y - b.h / 2);
    float x2 = std::min(a.x + a.w / 2, b.x + b.w / 2);
    float y2 = std::min(a.y + a.h / 2, b.y + b.h / 2);

    if (x2 < x1 || y2 < y1)
        return 0.0f;
    float intersection = (x2 - x1) * (y2 - y1);
    float area_a = a.w * a.h;
    float area_b = b.w * b.h;
    return intersection / (area_a + area_b - intersection);
}

std::vector<Detection> decode_output(const float *data, float conf_threshold)
{
    std::vector<Detection> boxes;
    int grid_area = GRID_W * GRID_H;

    for (int cy = 0; cy < GRID_H; ++cy)
    {
        for (int cx = 0; cx < GRID_W; ++cx)
        {
            for (int b = 0; b < NUM_ANCHORS; ++b)
            {
                int channel_base = b * BLOCK_SIZE;
                auto get = [&](int offset_c)
                {
                    return data[(channel_base + offset_c) * grid_area + cy * GRID_W + cx];
                };

                float objectness = sigmoid(get(4));
                // Optimization: Skip immediately if objectness is low
                if (objectness < conf_threshold)
                    continue;

                float max_class_score = -1.0f;
                int best_class = -1;

                for (int c = 0; c < NUM_CLASSES; ++c)
                {
                    float class_prob = get(5 + c);
                    if (class_prob > max_class_score)
                    {
                        max_class_score = class_prob;
                        best_class = c;
                    }
                }

                // Final Score = Objectness (Simplified for v2)
                float final_score = objectness;

                if (final_score > conf_threshold)
                {
                    float bx = (sigmoid(get(0)) + cx) / GRID_W;
                    float by = (sigmoid(get(1)) + cy) / GRID_H;
                    float bw = (ANCHORS[2 * b] * std::exp(get(2))) / GRID_W;
                    float bh = (ANCHORS[2 * b + 1] * std::exp(get(3))) / GRID_H;
                    boxes.push_back({bx, by, bw, bh, final_score, best_class, CLASS_NAMES[best_class]});
                }
            }
        }
    }

    // NMS
    std::sort(boxes.begin(), boxes.end(), [](const Detection &a, const Detection &b)
              { return a.confidence > b.confidence; });

    std::vector<Detection> clean_boxes;
    std::vector<bool> removed(boxes.size(), false);

    for (size_t i = 0; i < boxes.size(); ++i)
    {
        if (removed[i])
            continue;
        clean_boxes.push_back(boxes[i]);
        for (size_t j = i + 1; j < boxes.size(); ++j)
        {
            if (iou(boxes[i], boxes[j]) > 0.4f)
                removed[j] = true;
        }
    }
    return clean_boxes;
}

int main(int, char **)
{
    // 1. Setup Window
    if (!glfwInit())
        return 1;
    const char *glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

    GLFWwindow *window = glfwCreateWindow(1280, 720, "YOLOv2 Dashboard", NULL, NULL);
    if (window == NULL)
        return 1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // 2. Setup ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    (void)io;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // 3. Setup Webcam
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
        std::cerr << "Error: Could not open webcam." << std::endl;
        return -1;
    }

    // 4. Load Engine
    std::cout << "Loading Engine..." << std::endl;
    ModelLoader loader;
    if (!loader.load("tiny_yolo.onnx"))
    {
        // Fallback check
        if (!loader.load("../tiny_yolo.onnx"))
        {
            std::cerr << "Failed to load model!" << std::endl;
            return 1;
        }
    }
    InferenceEngine engine;
    engine.load_model(loader);
    std::cout << "Engine Ready." << std::endl;

    std::vector<int64_t> input_shape = {1, 3, 416, 416};
    Tensor input_tensor(DataType::FLOAT32, input_shape, "image");

    GLTexture webcam_tex;
    cv::Mat frame, rgb_frame;
    // This vector remembers boxes from previous frames
    std::vector<Detection> persistent_boxes;
    const int BOX_LIFETIME = 15; // How many frames a box lingers (approx 0.5 seconds at 30FPS)

    // --- Main Loop ---
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        // A. Capture
        cap >> frame;
        if (frame.empty())
            continue;

        // B. Update GUI Texture
        cv::cvtColor(frame, rgb_frame, cv::COLOR_BGR2RGB);
        webcam_tex.update(rgb_frame);

        // C. RUN AI (Synchronous - No threads, no crashes)
        // 1. Prepare Data
        std::vector<float> raw_input = preprocess(frame);
        std::memcpy(input_tensor.data<float>(), raw_input.data(), raw_input.size() * sizeof(float));

        // 2. Run Inference
        engine.run(input_tensor);

        // 3. Get Output
        Tensor &output = engine.get_output();
        const float *out_data = output.data<float>();

        // --- PROBE START ---
        float max_conf = -1.0f;
        int max_index = -1;
        int total_cells = 125 * 13 * 13;

        // Scan specifically for the "Objectness" channels
        // (Indexes: 4, 29, 54, 79, 104 in every 125-block)
        for (int i = 0; i < 13 * 13; ++i)
        { // For every grid cell
            for (int b = 0; b < 5; ++b)
            { // For every anchor
                int channel_offset = (b * 25 + 4) * (13 * 13);
                int idx = channel_offset + i;
                float conf = sigmoid(out_data[idx]);
                if (conf > max_conf)
                {
                    max_conf = conf;
                    max_index = idx;
                }
            }
        }
        // --- PROBE END ---
        // D. ImGui Drawing
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        {
            ImGui::Begin("YOLO Dashboard");

            // --- CONTROLS ---
            static float threshold = 0.20f; // Lower default threshold since we have smoothing now
            ImGui::SliderFloat("Confidence", &threshold, 0.01f, 1.0f);

            // --- DRAW IMAGE ---
            ImVec2 pos = ImGui::GetCursorScreenPos();
            if (webcam_tex.id != 0)
                ImGui::Image((void *)(intptr_t)webcam_tex.id, ImVec2(640, 480));

            // --- STABILIZER LOGIC START ---

            // 1. Get Fresh Detections from AI
            std::vector<Detection> fresh_detections = decode_output(out_data, threshold);

            // 2. Age the Old Boxes (Decrease Timer)
            for (auto &box : persistent_boxes)
            {
                box.timer--;
            }

            // 3. Merge Fresh Detections into Persistent Boxes
            for (const auto &new_det : fresh_detections)
            {
                bool matched = false;

                // Try to find a matching box in our history
                for (auto &old_box : persistent_boxes)
                {
                    // If IoU > 0.3, assume it's the same object
                    if (iou(new_det, old_box) > 0.3f && new_det.class_id == old_box.class_id)
                    {
                        // UPDATE the old box with new coordinates
                        old_box.x = new_det.x;
                        old_box.y = new_det.y;
                        old_box.w = new_det.w;
                        old_box.h = new_det.h;
                        old_box.confidence = new_det.confidence;
                        old_box.timer = BOX_LIFETIME; // RESET Timer (It lives!)
                        matched = true;
                        break;
                    }
                }

                // If no match found, this is a brand new object
                if (!matched)
                {
                    Detection det = new_det;
                    det.timer = BOX_LIFETIME; // Give it full life
                    persistent_boxes.push_back(det);
                }
            }

            // 4. Remove Dead Boxes (Timer <= 0)
            // This idiom efficiently deletes items from a vector
            persistent_boxes.erase(
                std::remove_if(persistent_boxes.begin(), persistent_boxes.end(),
                               [](const Detection &d)
                               { return d.timer <= 0; }),
                persistent_boxes.end());

            // --- STABILIZER LOGIC END ---

            // 5. Draw the PERSISTENT boxes (not the fresh ones)
            ImDrawList *draw_list = ImGui::GetWindowDrawList();
            float img_w = 640.0f;
            float img_h = 480.0f;

            for (const auto &det : persistent_boxes)
            {
                // Fade out color as timer runs out
                float alpha = (float)det.timer / BOX_LIFETIME;
                ImU32 color = IM_COL32(0, 255, 0, (int)(255 * alpha));
                ImU32 text_bg = IM_COL32(0, 0, 0, (int)(200 * alpha));
                ImU32 text_col = IM_COL32(255, 255, 255, (int)(255 * alpha));

                float center_x = pos.x + det.x * img_w;
                float center_y = pos.y + det.y * img_h;
                float width = det.w * img_w;
                float height = det.h * img_h;

                float left = center_x - width / 2.0f;
                float top = center_y - height / 2.0f;
                float right = center_x + width / 2.0f;
                float bottom = center_y + height / 2.0f;

                draw_list->AddRect(ImVec2(left, top), ImVec2(right, bottom), color, 3.0f);

                char label_buf[32];
                sprintf(label_buf, "%s %.2f", det.label.c_str(), det.confidence);
                draw_list->AddText(ImVec2(left, top - 20), text_col, label_buf);
            }

            ImGui::Text("Active Objects: %lu", persistent_boxes.size());
            ImGui::End();
        }

        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.45f, 0.55f, 0.60f, 1.00f);
        glClear(GL_COLOR_BUFFER_BIT);

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
