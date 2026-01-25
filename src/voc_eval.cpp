#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <map>
#include <numeric>
#include <iomanip>
#include <opencv2/opencv.hpp>

#include "engine.h"
#include "model_loader.h"

namespace fs = std::filesystem;

// --- CONFIG ---
const float IOU_THRESHOLD = 0.5f;
const float CONF_THRESHOLD = 0.005f;

// --- YOLO CONSTANTS ---
const float ANCHORS[10] = {1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52};
const int NUM_CLASSES = 20;

const char *CLASS_NAMES[] = {
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};

struct Box
{
    float x1, y1, x2, y2;
    float score;
    int class_id;
    bool used = false;
};

struct ImageRecord
{
    std::string id;
    std::vector<Box> gt_boxes;
    std::vector<Box> det_boxes;
};

inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

float calculate_iou(const Box& a, const Box& b) {
    float xx1 = std::max(a.x1, b.x1);
    float yy1 = std::max(a.y1, b.y1);
    float xx2 = std::min(a.x2, b.x2);
    float yy2 = std::min(a.y2, b.y2);

    float w = std::max(0.0f, xx2 - xx1);
    float h = std::max(0.0f, yy2 - yy1);
    float inter = w * h;

    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);

    // Union = Area A + Area B - Intersection
    float union_area = area_a + area_b - inter;
    
    // Avoid division by zero
    if (union_area <= 1e-6) return 0.0f;

    return inter / union_area;
}

// --- NEW: Letterbox Info Struct ---
struct LetterboxInfo
{
    float scale;
    int x_offset;
    int y_offset;
    int original_w;
    int original_h;
};

// --- NEW: Preprocess with Letterboxing ---
std::vector<float> preprocess_letterbox(const cv::Mat &src, LetterboxInfo &info)
{
    info.original_w = src.cols;
    info.original_h = src.rows;

    int target_w = 416;
    int target_h = 416;

    // 1. Calculate Scale (Min to fit)
    info.scale = std::min((float)target_w / src.cols, (float)target_h / src.rows);
    int new_w = (int)(src.cols * info.scale);
    int new_h = (int)(src.rows * info.scale);

    // 2. Resize
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h));

    // 3. Create Gray Canvas
    cv::Mat canvas(target_h, target_w, CV_8UC3, cv::Scalar(128, 128, 128));

    // 4. Center the image
    info.x_offset = (target_w - new_w) / 2;
    info.y_offset = (target_h - new_h) / 2;

    resized.copyTo(canvas(cv::Rect(info.x_offset, info.y_offset, new_w, new_h)));

    // 5. Convert to Float Planar (0-255 Scale)
    cv::cvtColor(canvas, canvas, cv::COLOR_BGR2RGB);

    std::vector<float> output;
    output.reserve(3 * 416 * 416);
    std::vector<cv::Mat> channels(3);
    cv::split(canvas, channels);

    for (int c = 0; c < 3; ++c)
    {
        for (int y = 0; y < 416; ++y)
        {
            for (int x = 0; x < 416; ++x)
            {
                output.push_back(static_cast<float>(channels[c].at<uint8_t>(y, x)));
            }
        }
    }
    return output;
}
// --- NEW: Non-Maximum Suppression ---
std::vector<Box> perform_nms(std::vector<Box> &boxes, float nms_threshold)
{
    if (boxes.empty())
        return {};

    // 1. Sort by Score (High to Low)
    std::sort(boxes.begin(), boxes.end(), [](const Box &a, const Box &b)
              { return a.score > b.score; });

    std::vector<Box> keep;
    std::vector<bool> suppressed(boxes.size(), false);

    for (size_t i = 0; i < boxes.size(); ++i)
    {
        if (suppressed[i])
            continue;
        keep.push_back(boxes[i]);

        for (size_t j = i + 1; j < boxes.size(); ++j)
        {
            if (suppressed[j])
                continue;

            // Only compare boxes of the same class
            if (boxes[i].class_id != boxes[j].class_id)
                continue;

            // Calculate IoU
            float xx1 = std::max(boxes[i].x1, boxes[j].x1);
            float yy1 = std::max(boxes[i].y1, boxes[j].y1);
            float xx2 = std::min(boxes[i].x2, boxes[j].x2);
            float yy2 = std::min(boxes[i].y2, boxes[j].y2);

            float w = std::max(0.0f, xx2 - xx1);
            float h = std::max(0.0f, yy2 - yy1);
            float inter = w * h;

            float area_i = (boxes[i].x2 - boxes[i].x1) * (boxes[i].y2 - boxes[i].y1);
            float area_j = (boxes[j].x2 - boxes[j].x1) * (boxes[j].y2 - boxes[j].y1);
            float union_area = area_i + area_j - inter;

            if (union_area > 0 && (inter / union_area) > nms_threshold)
            {
                suppressed[j] = true; // Kill the duplicate
            }
        }
    }
    return keep;
}

// --- UPDATED: Decode with Coordinate Correction ---
// Now takes 'LetterboxInfo' to map boxes back to original image
std::vector<Box> decode(const float *data, const LetterboxInfo &info)
{
    std::vector<Box> boxes;
    int grid_area = 13 * 13;

    for (int cy = 0; cy < 13; ++cy)
    {
        for (int cx = 0; cx < 13; ++cx)
        {
            for (int b = 0; b < 5; ++b)
            {
                int offset = (b * (5 + NUM_CLASSES)) * grid_area + cy * 13 + cx;
                auto get = [&](int i)
                { return data[offset + i * grid_area]; };

                float obj = sigmoid(get(4));
                if (obj < CONF_THRESHOLD)
                    continue;

                int best_c = 0;
                float max_c = -1;
                for (int c = 0; c < NUM_CLASSES; ++c)
                {
                    float val = get(5 + c);
                    if (val > max_c)
                    {
                        max_c = val;
                        best_c = c;
                    }
                }

                float score = obj; // Tiny YOLO v2 standard

                // 1. Get Box in 416x416 Canvas Coordinates
                float bx = (sigmoid(get(0)) + cx) / 13.0f * 416.0f;                  // Center X (pixels)
                float by = (sigmoid(get(1)) + cy) / 13.0f * 416.0f;                  // Center Y (pixels)
                float bw = (ANCHORS[2 * b] * std::exp(get(2))) / 13.0f * 416.0f;     // Width (pixels)
                float bh = (ANCHORS[2 * b + 1] * std::exp(get(3))) / 13.0f * 416.0f; // Height (pixels)

                // 2. Convert to Canvas Corners (x1, y1, x2, y2)
                float canvas_x1 = bx - bw / 2.0f;
                float canvas_y1 = by - bh / 2.0f;
                float canvas_x2 = bx + bw / 2.0f;
                float canvas_y2 = by + bh / 2.0f;

                // 3. CRITICAL: Map back to Original Image
                // Formula: original = (canvas - offset) / scale
                Box box;
                box.class_id = best_c;
                box.score = score;

                box.x1 = (canvas_x1 - info.x_offset) / info.scale;
                box.y1 = (canvas_y1 - info.y_offset) / info.scale;
                box.x2 = (canvas_x2 - info.x_offset) / info.scale;
                box.y2 = (canvas_y2 - info.y_offset) / info.scale;

                // Clamp to image boundaries
                box.x1 = std::max(0.0f, std::min((float)info.original_w, box.x1));
                box.y1 = std::max(0.0f, std::min((float)info.original_h, box.y1));
                box.x2 = std::max(0.0f, std::min((float)info.original_w, box.x2));
                box.y2 = std::max(0.0f, std::min((float)info.original_h, box.y2));

                boxes.push_back(box);
            }
        }
    }
    return boxes;
}

std::vector<Box> load_ground_truth(const std::string &path, int img_w, int img_h)
{
    std::vector<Box> boxes;
    std::ifstream file(path);
    if (!file.is_open())
        return boxes;

    int cls;
    float cx, cy, w, h;
    while (file >> cls >> cx >> cy >> w >> h)
    {
        Box b;
        b.class_id = cls;
        b.x1 = (cx - w / 2) * img_w;
        b.y1 = (cy - h / 2) * img_h;
        b.x2 = (cx + w / 2) * img_w;
        b.y2 = (cy + h / 2) * img_h;
        b.used = false;
        boxes.push_back(b);
    }
    return boxes;
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cerr << "Usage: ./voc_eval <model.onnx> <data_dir>" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::string data_dir = argv[2];
    std::string img_dir = data_dir + "/JPEGImages"; // Fixed Path
    std::string lbl_dir = data_dir + "/labels";

    ModelLoader loader;
    if (!loader.load(model_path))
        return 1;
    InferenceEngine engine;
    engine.load_model(loader);

    // Scan Files (JPG or jpg)
    std::vector<ImageRecord> records;
    for (const auto &entry : fs::directory_iterator(img_dir))
    {
        std::string ext = entry.path().extension().string();
        if (ext == ".jpg" || ext == ".JPG" || ext == ".jpeg")
        {
            ImageRecord rec;
            rec.id = entry.path().stem().string();
            records.push_back(rec);
        }
    }
    std::cout << "Found " << records.size() << " images." << std::endl;

    std::vector<int64_t> input_shape = {1, 3, 416, 416};
    Tensor input_tensor(DataType::FLOAT32, input_shape, "input");

    int processed = 0;
    for (auto &rec : records)
    {
        cv::Mat img = cv::imread(img_dir + "/" + rec.id + ".jpg");
        if (img.empty())
            img = cv::imread(img_dir + "/" + rec.id + ".JPG"); // Try uppercase
        if (img.empty())
            continue;

        // B. Load GT
        rec.gt_boxes = load_ground_truth(lbl_dir + "/" + rec.id + ".txt", img.cols, img.rows);

        // C. Run AI with Letterbox
        LetterboxInfo info;
        std::vector<float> raw = preprocess_letterbox(img, info); // Use new function

        std::memcpy(input_tensor.data<float>(), raw.data(), raw.size() * sizeof(float));
        engine.run(input_tensor);

        // D. Decode with Mapping
        std::vector<Box> raw_boxes = decode(engine.get_output().data<float>(), info);

        // Use 0.45 as threshold (Standard for YOLOv2)
        rec.det_boxes = perform_nms(raw_boxes, 0.45f);

        processed++;
        if (processed % 100 == 0)
            std::cout << "." << std::flush;
    }
    std::cout << "\nInference Complete. Computing mAP..." << std::endl;

    // --- mAP Calculation ---
    float map_sum = 0.0f;
    std::cout << "\n--- PER CLASS RESULTS (AP @ IoU=0.5) ---" << std::endl;
    std::cout << std::left << std::setw(15) << "Class" << "AP" << std::endl;
    std::cout << "----------------------" << std::endl;

    for (int c = 0; c < NUM_CLASSES; ++c)
    {
        std::vector<std::pair<float, bool>> predictions;
        int total_gt = 0;

        for (auto &rec : records)
        {
            std::vector<Box *> gt_c;
            for (auto &gt : rec.gt_boxes)
            {
                if (gt.class_id == c)
                {
                    gt_c.push_back(&gt);
                    gt.used = false;
                    total_gt++;
                }
            }

            std::vector<Box> det_c;
            for (const auto &det : rec.det_boxes)
            {
                if (det.class_id == c)
                    det_c.push_back(det);
            }
            std::sort(det_c.begin(), det_c.end(), [](const Box &a, const Box &b)
                      { return a.score > b.score; });

            for (const auto &det : det_c)
            {
                float best_iou = 0.0f;
                Box *best_gt = nullptr;
                for (auto *gt : gt_c)
                {
                    float iou = calculate_iou(det, *gt);
                    if (iou > best_iou)
                    {
                        best_iou = iou;
                        best_gt = gt;
                    }
                }
                if (best_iou >= IOU_THRESHOLD && best_gt && !best_gt->used)
                {
                    predictions.push_back({det.score, true});
                    best_gt->used = true;
                }
                else
                {
                    predictions.push_back({det.score, false});
                }
            }
        }

        std::sort(predictions.begin(), predictions.end(), [](auto &a, auto &b)
                  { return a.first > b.first; });

        float ap = 0.0f;
        if (total_gt > 0)
        {
            std::vector<float> precs, recalls;
            int tp = 0;
            for (size_t i = 0; i < predictions.size(); ++i)
            {
                if (predictions[i].second)
                    tp++;
                precs.push_back((float)tp / (i + 1));
                recalls.push_back((float)tp / total_gt);
            }
            for (float t = 0.0f; t <= 1.0f; t += 0.1f)
            {
                float p_max = 0.0f;
                for (size_t i = 0; i < recalls.size(); ++i)
                {
                    if (recalls[i] >= t)
                        p_max = std::max(p_max, precs[i]);
                }
                ap += p_max / 11.0f;
            }
        }
        std::cout << std::left << std::setw(15) << CLASS_NAMES[c] << std::fixed << std::setprecision(4) << ap << std::endl;
        map_sum += ap;
    }

    std::cout << "----------------------" << std::endl;
    std::cout << "mAP (Mean Average Precision): " << (map_sum / NUM_CLASSES) << std::endl;

    return 0;
}
