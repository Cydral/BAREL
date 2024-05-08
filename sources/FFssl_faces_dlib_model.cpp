#include <iostream>
#include <string>

#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/range/algorithm/count.hpp>
#include <boost/tokenizer.hpp>

#include <dlib/cmd_line_parser.h>
#include <dlib/data_io.h>
#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>

#include "get_lfw_pairs.h"

using namespace std;
using namespace dlib;
namespace fs = boost::filesystem;

string training_filename = "dlib_face_recognition_resnetssl_training.dat";
string default_filename = "dlib_face_recognition_resnet_model_v2.dat";
string model_training_filename = boost::filesystem::current_path().string() + "/" + training_filename;
string default_model_filename = boost::filesystem::current_path().string() + "/" + default_filename;

const size_t default_face_size = 150;
const size_t default_batch_size = 64;
const size_t default_pipe_size = 2 * default_batch_size;

namespace model {
    template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
    using residual = add_prev1<block<N, BN, 1, tag1<SUBNET>>>;

    template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
    using residual_down = add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<block<N, BN, 2, tag1<SUBNET>>>>>>;

    template <int N, template <typename> class BN, int stride, typename SUBNET>
    using block = BN<con<N, 3, 3, 1, 1, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;

    template <int N, typename SUBNET> using res = relu<residual<block, N, bn_con, SUBNET>>;
    template <int N, typename SUBNET> using ares = relu<residual<block, N, affine, SUBNET>>;
    template <int N, typename SUBNET> using res_down = relu<residual_down<block, N, bn_con, SUBNET>>;
    template <int N, typename SUBNET> using ares_down = relu<residual_down<block, N, affine, SUBNET>>;

    template <typename SUBNET> using level0 = res_down<256, SUBNET>;
    template <typename SUBNET> using level1 = res<256, res<256, res_down<256, SUBNET>>>;
    template <typename SUBNET> using level2 = res<128, res<128, res_down<128, SUBNET>>>;
    template <typename SUBNET> using level3 = res<64, res<64, res<64, res_down<64, SUBNET>>>>;
    template <typename SUBNET> using level4 = res<32, res<32, res<32, SUBNET>>>;

    template <typename SUBNET> using alevel0 = ares_down<256, SUBNET>;
    template <typename SUBNET> using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
    template <typename SUBNET> using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
    template <typename SUBNET> using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
    template <typename SUBNET> using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

    template <typename SUBNET> using ssl_projector = fc<128, relu<bn_fc<fc<512, avg_pool_everything<SUBNET>>>>>;
    template <typename SUBNET> using metric_projector = fc_no_bias<128, avg_pool_everything<SUBNET>>;
    using train_ssl = loss_barlow_twins<ssl_projector<level0<level1<level2<level3<level4<max_pool<3, 3, 2, 2, relu<bn_con<con<32, 7, 7, 2, 2, input_rgb_image_pair>>>>>>>>>>>;
    using feats_ssl = loss_metric<ssl_projector<alevel0<alevel1<alevel2<alevel3<alevel4<max_pool<3, 3, 2, 2, relu<affine<con<32, 7, 7, 2, 2, input_rgb_image_sized<default_face_size>>>>>>>>>>>>;
    using train_std = loss_metric<metric_projector<level0<level1<level2<level3<level4<max_pool<3, 3, 2, 2, relu<bn_con<con<32, 7, 7, 2, 2, input_rgb_image_sized<default_face_size>>>>>>>>>>>>;
    using feats_std = loss_metric<metric_projector<alevel0<alevel1<alevel2<alevel3<alevel4<max_pool<3, 3, 2, 2, relu<affine<con<32, 7, 7, 2, 2,input_rgb_image_sized<default_face_size>>>>>>>>>>>>;
}

// Declares an atomic boolean variable g_interrupted and a console event handler ConsoleHandler
// that sets g_interrupted to true on certain events.
atomic_bool g_interrupted = false;
BOOL WINAPI ConsoleHandler(DWORD CEvent) {
    switch (CEvent)
    {
    case CTRL_C_EVENT:
        g_interrupted = true;
        break;
    case CTRL_BREAK_EVENT:
        break;
    case CTRL_CLOSE_EVENT:
        g_interrupted = true;
        break;
    case CTRL_LOGOFF_EVENT:
        g_interrupted = true;
        break;
    case CTRL_SHUTDOWN_EVENT:
        g_interrupted = true;
        break;
    }
    return TRUE;
}

// Resizes an input matrix to a specified size in-place.
template <typename pixel_type>
void resize_inplace(matrix<pixel_type>& inout, long size = default_face_size) {
    if (inout.nr() != size || inout.nc() != size) {
        matrix<pixel_type> mem_img;
        mem_img.set_size(size, size);
        resize_image(inout, mem_img);
        inout = mem_img;
    }
}

// Generates a random cropping rectangle based on an input image and a random number generator.
// The rectangle's size is determined by a randomly sampled scale and aspect ratio, and the final position
// is randomly offset within the image bounds.
rectangle make_random_cropping_rect(const matrix<rgb_pixel>& image, dlib::rand& rnd) {
    const double mins = 9. / 16.;
    const double maxs = 9. / 10.;
    const auto scale = rnd.get_double_in_range(mins, maxs);
    const auto size = scale * std::min(image.nr(), image.nc());
    const rectangle rect(size, size);
    const point offset(rnd.get_random_32bit_number() % (image.nc() - rect.width()),
        rnd.get_random_32bit_number() % (image.nr() - rect.height()));
    return move_rect(rect, offset);
}

// Helper functions to generate different kinds of augmentations
matrix<rgb_pixel> std_augmentation(const matrix<rgb_pixel>& image, dlib::rand& rnd) {
    matrix<rgb_pixel> augmented = (rnd.get_random_double() < 0.4) ? jitter_image(image, rnd) : image;
    matrix<rgb_pixel> blurred;
    if (rnd.get_random_double() < 0.2) {
        matrix<rgb_pixel> crop;
        const auto rect = gaussian_blur(augmented, blurred, rnd.get_double_in_range(0.1, 1.1));
        extract_image_chip(blurred, rect, crop);
        blurred = crop;
    } else blurred = augmented;
    resize_inplace(blurred);
    if (rnd.get_random_double() < 0.6) disturb_colors(blurred, rnd, rnd.get_double_in_range(0.45, 0.65), rnd.get_double_in_range(0.45, 0.65));
    if (rnd.get_random_double() < 0.2) {
        matrix<unsigned char> gray;
        assign_image(gray, blurred);
        assign_image(blurred, gray);
    }
    return blurred;
}
matrix<rgb_pixel> ssl_augmentation(const matrix<rgb_pixel>& image, const bool prime, dlib::rand& rnd) {
    matrix<rgb_pixel> crop, blurred;
    if (!prime || (prime && rnd.get_random_double() < 0.2)) {
        const auto rect = gaussian_blur(image, blurred, rnd.get_double_in_range(0.1, 1.1));
        extract_image_chip(blurred, rect, crop);
        blurred = crop;
    } else blurred = image;

    const auto rect = make_random_cropping_rect(image, rnd);
    extract_image_chip(blurred, chip_details(rect, chip_dims(default_face_size, default_face_size)), crop);
    if (rnd.get_random_double() < 0.4) flip_image_left_right(crop);
    if (rnd.get_random_double() < 0.6) disturb_colors(crop, rnd, rnd.get_double_in_range(0.45, 0.65), rnd.get_double_in_range(0.45, 0.65));
    if (rnd.get_random_double() < 0.2) {
        matrix<unsigned char> gray;
        assign_image(gray, crop);
        assign_image(crop, gray);
    }    
    return crop;
}

// Finds the best facial detection within an image, extracts a face chip based on shape information, and returns it.
matrix<rgb_pixel> get_lfw_face_chip(const matrix<rgb_pixel>& in, const rectangle& rect, frontal_face_detector& face_extractor, shape_predictor& sp) {
    rectangle best_det;
    double best_overlap = 0;
    for (auto det : face_extractor(in, -0.9)) {
        auto overlap = box_intersection_over_union(rect, det);
        if (overlap > best_overlap) {
            best_det = det;
            best_overlap = overlap;
        }
    }
    if (best_overlap < 0.3) best_det = rect;
    auto shape = sp(in, best_det);
    matrix<rgb_pixel> face_chip;
    extract_image_chip(in, get_face_chip_details(shape, default_face_size, 0.25), face_chip);
    return face_chip;
}

// Applies quick jittering to an input image by generating multiple random crops using jitter_image and returns them as a vector.
std::vector<matrix<rgb_pixel>> quick_jitter_image(const matrix<rgb_pixel>& img, const long nb_iter) {
    thread_local dlib::rand rnd;
    std::vector<matrix<rgb_pixel>> crops;
    for (int i = 0; i < nb_iter; ++i) crops.push_back(jitter_image(img, rnd));
    return crops;
}

class progress_bar {
    static const auto overhead = sizeof " [100%]";
    std::ostream& os;
    const size_t bar_width;
    string message;
    const string full_bar;

public:
    progress_bar(std::ostream& os, string message_, size_t line_width = 50u, const char symbol = '.') : os{ os },
        bar_width{ line_width - overhead },
        message{ std::move(message_) },
        full_bar{ string(bar_width, symbol) + string(bar_width, ' ') } {
        if (message.size() + 1 >= bar_width || message.find('\n') != message.npos) {
            os << message << '\n';
            message.clear();
        }
        else {
            message += ' ';
        }
        write(0, 100);
    }
    progress_bar(const progress_bar&) = delete;
    progress_bar& operator=(const progress_bar&) = delete;
    ~progress_bar() {
        write(100, 100);
        os << '\n';
    }

    void write(size_t i, size_t n);
};
void progress_bar::write(size_t i, size_t n) {
    if (i % 10 != 0) return;
    double fraction = (double)i / (double)n;
    if (fraction < 0) fraction = 0;
    else if (fraction > 1) fraction = 1;

    auto width = bar_width - message.size();
    auto offset = bar_width - static_cast<unsigned>(width * fraction);

    os << '\r' << message;
    os.write(full_bar.data() + offset, width);
    os << " [" << std::setw(3) << static_cast<int>(100 * fraction) << "%] " << std::flush;
}

struct my_file_info {
    matrix<float, 0, 1> signature;
    std::string parent_dir;

    my_file_info(const matrix<float, 0, 1>& sig, const std::string& parent)
        : signature(sig), parent_dir(parent) {}
};
class file_list {
public:
    void add_file(const std::string& filename, const matrix<float, 0, 1>& signature) {
        std::size_t last_slash = filename.find_last_of("/");
        if (last_slash == std::string::npos) last_slash = filename.find_last_of("\\");
        std::string parent_dir = (last_slash != std::string::npos) ? filename.substr(0, last_slash) : filename;
        files.emplace_back(signature, parent_dir);
    }
    const std::vector<my_file_info>& get_files() const { return files; }
    void display() const {
        for (const auto& file_info : files) {
            std::cout << "Parent Directory: " << file_info.parent_dir << std::endl;
            std::cout << "Signature: [ " << trans(file_info.signature) << "]\n\n";
        }
    }
    const my_file_info& operator[](size_t index) const {
        if (index < files.size()) return files[index];
        else throw std::out_of_range("Index out of bounds");
    }

private:
    std::vector<my_file_info> files;
};

std::vector<std::vector<string>> load_people_list(const string& dir) {
    const unsigned long minimum_samples_per_object = 5;
    std::vector<std::vector<string>> objects;
    for (auto subdir : directory(dir).get_dirs()) {
        std::vector<string> imgs;
        for (auto img : subdir.get_files()) {
            string filename = img;
            imgs.push_back(filename);
        }
        if (imgs.size() >= minimum_samples_per_object) objects.push_back(imgs);
    }
    return objects;
}
bool load_mini_batch(const size_t num_people, const size_t samples_per_id, dlib::rand& rnd, const std::vector<std::vector<string>>& objs, std::vector<matrix<rgb_pixel>>& images, std::vector<unsigned long>& labels, const bool for_training = true) {
    images.clear();
    labels.clear();
    std::vector<bool> already_selected(objs.size(), false);
    matrix<rgb_pixel> image;
    for (size_t i = 0; i < num_people; ++i) {
        size_t id = rnd.get_random_32bit_number() % objs.size();
        while (already_selected[id]) id = rnd.get_random_32bit_number() % objs.size();
        already_selected[id] = true;
        for (size_t j = 0; j < samples_per_id; ++j) {
            const auto& obj = objs[id][rnd.get_random_32bit_number() % objs[id].size()];
            try { load_image(image, obj); }
            catch (...) {
                cerr << "Error during image loading: " << obj << endl;
                continue;
            }
            resize_inplace(image);
            images.push_back(std::move(image));
            labels.push_back(id);
        }
    }
    for (auto&& aug : images) aug = std_augmentation(aug, rnd);
    return (images.size() > 0);
}

bool find_top_rank(const file_list& benchmark_list, const size_t pos, size_t top_rank) {
    const auto& src_desc = benchmark_list[pos].signature;
    const auto& src_class = benchmark_list[pos].parent_dir;
    std::vector<std::pair<double, std::string>> scores;
    size_t total_files = benchmark_list.get_files().size();
    for (size_t cur_pos = 0; cur_pos < total_files; ++cur_pos) {
        if (cur_pos != pos) scores.emplace_back(length(src_desc - benchmark_list[cur_pos].signature), benchmark_list[cur_pos].parent_dir);
        cur_pos++;
    }
    std::sort(scores.begin(), scores.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    bool found = false;
    for (size_t i = 0; i < __min(top_rank, scores.size()) && !found; ++i) {
        if (scores[i].second == src_class) found = true;
    }
    return found;
}

// SSL TRAINING PROCESS
void ssl_training(const std::vector<file>& images, const std::string& model_filename, const bool sync_file, const double learning_rate, const long dims, const double lambda, const size_t batch_size, const long patience, std::vector<int>& gpus) {
    const string model_sync_filename = boost::filesystem::current_path().string() + "/ssl_trainer_state.dat";
    set_dnn_prefer_smallest_algorithms();
    model::train_ssl net((loss_barlow_twins_(lambda)));
    disable_duplicative_biases(net);
    
    if (!fs::exists(model_sync_filename) && fs::exists(model_training_filename)) deserialize(model_training_filename) >> net;
    if (fs::exists(model_training_filename)) {
        boost::filesystem::path destinationPath = model_training_filename;
        boost::filesystem::path backupPath = model_training_filename;
        backupPath.replace_extension(".bak");
        boost::filesystem::copy_file(destinationPath, backupPath, boost::filesystem::copy_option::overwrite_if_exists);
    }
    dnn_trainer<model::train_ssl> trainer(net, sgd(0.0001, 0.9), gpus);
    trainer.set_mini_batch_size(batch_size);
    trainer.set_learning_rate(learning_rate);
    trainer.set_min_learning_rate(1e-5);
    trainer.set_iterations_without_progress_threshold(patience);
    if (sync_file) trainer.set_synchronization_file(model_sync_filename, std::chrono::minutes(30));
    trainer.be_verbose();
    set_all_bn_running_stats_window_sizes(net, 2000);

    // Output training parameters
    std::cout << "The network has " << net.num_layers << " layers in it." << std::endl;
    std::cout << net << std::endl;
    std::cout << std::endl << trainer << std::endl;

    // Main thread to load dataset
    dlib::pipe<std::vector<pair<matrix<rgb_pixel>, matrix<rgb_pixel>>>> images_pipeline(default_pipe_size);
    auto t_lambda = [&images_pipeline, &images, &batch_size](time_t seed) {
        dlib::rand rnd(time(0) + seed);
        matrix<rgb_pixel> img;
        std::vector<pair<matrix<rgb_pixel>, matrix<rgb_pixel>>> local_images;        
        while (images_pipeline.is_enabled()) {
            for (size_t i = 0; i < batch_size && !g_interrupted; ++i) {
                auto image_info = images[rnd.get_random_32bit_number() % images.size()];
                try { load_image(img, image_info.full_name()); }
                catch (...) { continue; }                
                local_images.emplace_back(ssl_augmentation(img, false, rnd), ssl_augmentation(img, true, rnd));
            }
            images_pipeline.enqueue(local_images);
            local_images.clear();
        }
    };
    uint32_t nb_threads = (std::thread::hardware_concurrency() / 2);
    std::vector<std::thread> threads;
    for (uint32_t t = 0; t < nb_threads; t++) threads.push_back(std::thread([&t, t_lambda]() { t_lambda(t + 1); }));

    // For performance reason, we wait a bit to have a complete pipe before learning...
    std::cout << "Waiting for the initial pipe loading... ";
    while (images_pipeline.size() < default_pipe_size) std::this_thread::sleep_for(std::chrono::seconds(3));
    std::cout << "done" << std::endl;
    // ... and than we can start the learning process
    std::vector<pair<matrix<rgb_pixel>, matrix<rgb_pixel>>> batch;
    while (trainer.get_learning_rate() >= trainer.get_min_learning_rate() && !g_interrupted) {
        batch.clear();
        images_pipeline.dequeue(batch);
        trainer.train_one_step(batch);
    }

    // Training done, tell threads to stop and make sure to wait for them to finish before moving on.
    images_pipeline.disable();
    for (std::thread& t : threads) if (t.joinable()) t.join();

    // Also wait for threaded processing to stop in the trainer.
    std::cout << "Saving network" << std::endl;
    trainer.get_net();
    net.clean();    
    serialize(model_training_filename) << net;
    model::train_std fnet;
    auto backbone = std::move(net.subnet().subnet().subnet().subnet().subnet());
    fnet.subnet().subnet() = backbone;
    serialize(model_filename) << fnet;
    if (!g_interrupted) {
        fs::remove(model_sync_filename);
        fs::remove((string(model_sync_filename) + string("_")).c_str());
    }
}
// FINETUNING PROCESS FOR STANDARD RESNET
void metric_training(const std::string& folders, const std::string& model_filename, const bool sync_file, const double learning_rate, const size_t batch_size, const double weight_decay, const double momentum, const long patience, const long freezing_level, std::vector<int>& gpus) {
    if (!fs::is_directory(folders)) return;
    auto objs = load_people_list(folders);

    const string model_sync_filename = boost::filesystem::current_path().string() + "/metric_trainer_state.dat";
    set_dnn_prefer_smallest_algorithms();
    model::train_std net;
    dlib::disable_duplicative_biases(net);

    if (fs::exists(model_filename)) {
        boost::filesystem::path destinationPath = model_filename;
        boost::filesystem::path backupPath = model_filename;
        backupPath.replace_extension(".bak");
        boost::filesystem::copy_file(destinationPath, backupPath, boost::filesystem::copy_option::overwrite_if_exists);
        deserialize(model_filename) >> net;
    }
    dnn_trainer<model::train_std> trainer(net, sgd(weight_decay, momentum), gpus);
    trainer.set_mini_batch_size(batch_size);
    trainer.set_learning_rate(learning_rate);
    trainer.set_min_learning_rate(1e-5);
    trainer.set_learning_rate_shrink_factor(0.1);
    trainer.set_iterations_without_progress_threshold(patience);
    if (sync_file) trainer.set_synchronization_file(model_sync_filename, std::chrono::minutes(30));
    trainer.be_verbose();
    set_all_bn_running_stats_window_sizes(net, 2000);

    // "Freeze" the layers according to the freezing level - Use "0" to unfreeze all layers
    set_all_learning_rate_multipliers(net, 1);
    if (freezing_level != 0) {
        switch (freezing_level) {
        case 1:            
            set_learning_rate_multipliers_range<3, 131>(net, 1e-6); // Only fine-tune the top (last) FC layer
            break;
        case 2:
            set_learning_rate_multipliers_range<41, 131>(net, 1e-6); // FC and layer_0 layers
            break;
        case 3:
            set_learning_rate_multipliers_range<68, 131>(net, 1e-6); // FC and layer_0_1 layers
            break;
        case 4:
            set_learning_rate_multipliers_range<103, 131>(net, 1e-6); // FC and layer_0_1_2 layers
            break;
        default:
            set_learning_rate_multipliers_range<3, 131>(net, 1e-6);
        }
    }

    // Output training parameters
    std::cout << "The network has " << net.num_layers << " layers in it." << std::endl;
    std::cout << net << std::endl;
    std::cout << std::endl << trainer << std::endl;

    // Main thread to load dataset
    dlib::pipe<std::vector<matrix<rgb_pixel>>> images_pipeline(default_pipe_size);
    dlib::pipe<std::vector<unsigned long>> labels_pipeline(default_pipe_size);
    auto t_lambda = [&images_pipeline, &labels_pipeline, &objs](time_t seed) {
        dlib::rand rnd(time(0) + seed);
        std::vector<matrix<rgb_pixel>> images;
        std::vector<unsigned long> labels;
        while (images_pipeline.is_enabled()) {
            try {
                if (load_mini_batch(6, 5, rnd, objs, images, labels)) {
                    images_pipeline.enqueue(images);
                    labels_pipeline.enqueue(labels);
                }
            }
            catch (std::exception& e) {
                std::cout << "Exception in loading image in <t_lambda>: " << e.what() << std::endl;
            }
        }
    };
    uint32_t nb_threads = (std::thread::hardware_concurrency() / 2) + 1;
    std::vector<std::thread> threads;
    for (uint32_t t = 0; t < nb_threads; t++) threads.push_back(std::thread([&t, t_lambda]() { t_lambda(t + 1); }));

    // For performance reason, we wait a bit to have a complete pipe before learning...
    std::cout << "Waiting for the initial pipe loading... ";
    while (images_pipeline.size() < default_pipe_size) std::this_thread::sleep_for(std::chrono::seconds(1));
    std::cout << "done" << std::endl;
    // ... and than we can start the learning process
    std::vector<matrix<rgb_pixel>> images;
    std::vector<unsigned long> labels;
    while (trainer.get_learning_rate() >= trainer.get_min_learning_rate() && !g_interrupted) {
        images_pipeline.dequeue(images);
        labels_pipeline.dequeue(labels);
        trainer.train_one_step(images.begin(), images.end(), labels.begin());
    }

    // Training done, tell threads to stop and make sure to wait for them to finish before moving on.
    images_pipeline.disable();
    labels_pipeline.disable();
    for (std::thread& t : threads) if (t.joinable()) t.join();

    // Also wait for threaded processing to stop in the trainer.
    trainer.get_net();
    net.clean();
    set_all_learning_rate_multipliers(net, 1);
    std::cout << "Saving network" << std::endl;
    serialize(model_filename) << net;
    if (!g_interrupted) {
        fs::remove(model_sync_filename);
        fs::remove((string(model_sync_filename) + string("_")).c_str());
    }
}

int main(const int argc, const char** argv)
try
{
    SetConsoleCtrlHandler((PHANDLER_ROUTINE)ConsoleHandler, TRUE);
    command_line_parser parser;
    parser.add_option("train", "start the internal training process (arg: folders)", 1);
    parser.add_option("fine-tune", "refine the pre-trained model (arg: folders)", 1);
    parser.add_option("std-benchmark", "perform a standard search benchmark (arg: folders)", 1);
    parser.add_option("lfw-benchmark", "assess against the LFW benchmark (arg: folders)", 1);
    parser.add_option("model-filename", "set the model filename (default: dlib_face_recognition_resnet_model_v2.dat)", 1);
    parser.add_option("gpus", "set the gpu list (e.g. \"0,1,2\" - default: 0)", 1);
    parser.add_option("use-sync-file", "use sync file during training (default: none)");
    parser.add_option("batch", "set the mini batch size (default: 64)", 1);
    parser.add_option("lambda", "penalize off-diagonal terms (default: 1/dims)", 1);
    parser.add_option("weight-decay", "set the weight decay value (default: 0.0004)", 1);
    parser.add_option("momentum", "set the momentum value (default: 0.9)", 1);
    parser.add_option("learning-rate", "set the initial learning rate (default: 1e-1)", 1);
    parser.add_option("threshold", "set the distance threshold value (default: 0.6)", 1);
    parser.add_option("jittering", "set the benchmark jittering value (default: 10)", 1);
    parser.add_option("patience", "set the training patience value (default: 25000)", 1);
    parser.add_option("freezing-level", "set the model freezing level (default: 1)", 1);
    parser.set_group_name("Help Options");
    parser.add_option("h", "alias for --help");
    parser.add_option("help", "display this message and exit");
    parser.parse(argc, argv);
    if (parser.option("h") || parser.option("help")) {
        parser.print_options();
        return EXIT_SUCCESS;
    }
    size_t num_gpus = 1;
    std::vector<int> gpus = { 0 };
    const string gpu_list = get_option(parser, "gpus", "");
    if (!gpu_list.empty()) {
        gpus.clear();
        boost::tokenizer<boost::escaped_list_separator<char>> tokens(gpu_list, boost::escaped_list_separator<char>());
        for (const auto& token : tokens) gpus.push_back(stoi(token));
        num_gpus = gpus.size();
        std::cout << ">>> Using " << num_gpus << " GPU" << (num_gpus > 0 ? "s" : "") << ": { ";
        for (const auto& g : gpus) std::cout << g << " ";
        std::cout << "} <<<" << std::endl;
    }    

    if (parser.option("train")) {
        const string model_filename = get_option(parser, "model-filename", default_model_filename);
        const bool use_sync_file = (parser.option("use-sync-file").count() != 0);
        const size_t batch_size = get_option(parser, "batch", default_batch_size);
        const long dims = 128;
        const double lambda = get_option(parser, "lambda", 1.0 / dims);
        const double learning_rate = get_option(parser, "learning-rate", 1e-1);
        const string root_folders = get_option(parser, "train", fs::current_path().string());
        const long patience = get_option(parser, "patience", 25000);

        std::cout << "all used parameters:" << endl;
        std::cout << "\t- number of GPUs: " << num_gpus << endl;
        std::cout << "\t- batch size: " << batch_size << endl;
        std::cout << "\t- dimension: " << dims << endl;
        std::cout << "\t- lambda: " << lambda << endl;
        std::cout << "\t- learning rate: " << learning_rate << endl;        
        std::cout << "\t- patience: " << patience << endl;
        std::cout << "\t- folders: " << root_folders << endl << endl;

        // Start the training process
        const std::vector<file> training_images = dlib::get_files_in_directory_tree(root_folders, dlib::match_endings(".jpeg .jpg .png"));
        cout << "images in dataset: " << training_images.size() << endl;
        if (training_images.size() == 0) {
            cout << "Didn't find images for the feature extraction process." << endl;
            return EXIT_FAILURE;
        }
        ssl_training(training_images, model_filename, use_sync_file, learning_rate, dims, lambda, batch_size, patience, gpus);
    } else if (parser.option("std-benchmark")) {
        const string folders = get_option(parser, "std-benchmark", fs::current_path().string());
        const string model_filename = get_option(parser, "model-filename", default_model_filename);
        const long jittering = get_option(parser, "jittering", 10);
        const double threshold = get_option(parser, "threshold", 0.6);
        const std::vector<file> files = dlib::get_files_in_directory_tree(folders, dlib::match_endings(".jpeg .jpg .png"));
        cout << "images in folders: " << files.size() << endl;
        if (files.size() == 0) {
            cout << "Didn't find images for the clustering process." << endl;
            return EXIT_FAILURE;
        }        
        model::feats_ssl f_ssl;
        model::feats_std f_ftn;
        const bool use_ssl_model = (model_filename.find(training_filename) != std::string::npos);        
        if (use_ssl_model) {
            model::train_ssl anet;
            if (fs::exists(model_filename)) deserialize(model_filename) >> anet;
            else {
                cerr << "Model file <" << model_filename << " not found in the current/given folder" << endl;
                return EXIT_SUCCESS;
            }
            f_ssl.subnet() = anet.subnet();
        } else {
            if (fs::exists(model_filename)) deserialize(model_filename) >> f_ftn;
            else {
                cerr << "Model file <" << model_filename << " not found in the current/given folder" << endl;
                return EXIT_SUCCESS;
            }
        }
        
        file_list benchmark_list;
        matrix<rgb_pixel> input_image;
        cout << "extracting features... ";
        for (size_t i = 0; i < files.size() && !g_interrupted; ++i) {
            try { load_image(input_image, files[i].full_name()); }
            catch (...) {
                cerr << "Error during image loading: " << files[i].full_name() << endl;
                continue;
            }
            resize_inplace(input_image);
            matrix<float, 0, 1> descriptor;
            if (use_ssl_model) {
                descriptor = mean(mat(f_ssl(quick_jitter_image(input_image, jittering))));
            } else {
                descriptor = mean(mat(f_ftn(quick_jitter_image(input_image, jittering))));
            }
            benchmark_list.add_file(files[i].full_name(), descriptor);
        }
        cout << "done" << endl;
        size_t num_right_t1 = 0, num_wrong_t1 = 0, num_right_t3 = 0, num_wrong_t3 = 0, num_right_t5 = 0, num_wrong_t5 = 0, total_files = benchmark_list.get_files().size();
        for (size_t cur_pos = 0; cur_pos < total_files && !g_interrupted; ++cur_pos) {
            if (find_top_rank(benchmark_list, cur_pos, 1)) ++num_right_t1;
            else ++num_wrong_t1;
            if (find_top_rank(benchmark_list, cur_pos, 3)) ++num_right_t3;
            else ++num_wrong_t3;
            if (find_top_rank(benchmark_list, cur_pos, 5)) ++num_right_t5;
            else ++num_wrong_t5;
        }
        cout << "*** statistics for top-1 searches" << endl;
        cout << "\tnum right:  " << num_right_t1 << endl;
        cout << "\tnum wrong:  " << num_wrong_t1 << endl;
        cout << "\taccuracy:   " << num_right_t1 / static_cast<float>(num_right_t1 + num_wrong_t1) << endl;
        cout << "\terror rate: " << num_wrong_t1 / static_cast<float>(num_right_t1 + num_wrong_t1) << endl;
        cout << "*** statistics for top-3 searches" << endl;
        cout << "\tnum right:  " << num_right_t3 << endl;
        cout << "\tnum wrong:  " << num_wrong_t3 << endl;
        cout << "\taccuracy:   " << num_right_t3 / static_cast<float>(num_right_t3 + num_wrong_t3) << endl;
        cout << "\terror rate: " << num_wrong_t3 / static_cast<float>(num_right_t3 + num_wrong_t3) << endl;
        cout << "*** statistics for top-5 searches" << endl;
        cout << "\tnum right:  " << num_right_t5 << endl;
        cout << "\tnum wrong:  " << num_wrong_t5 << endl;
        cout << "\taccuracy:   " << num_right_t5 / static_cast<float>(num_right_t5 + num_wrong_t5) << endl;
        cout << "\terror rate: " << num_wrong_t5 / static_cast<float>(num_right_t5 + num_wrong_t5) << endl;
    } else if (parser.option("fine-tune")) {
        const string model_filename = get_option(parser, "model-filename", default_model_filename);
        const bool use_sync_file = (parser.option("use-sync-file").count() != 0);
        const size_t batch_size = get_option(parser, "batch", default_batch_size);
        const double learning_rate = get_option(parser, "learning-rate", 1e-1);
        const double weight_decay = get_option(parser, "weight-decay", 0.0004);
        const double momentum = get_option(parser, "momentum", 0.9);
        const string root_folders = get_option(parser, "fine-tune", fs::current_path().string());
        const long patience = get_option(parser, "patience", 25000);
        const long freezing_level = get_option(parser, "freezing-level", 1);

        std::cout << "all used parameters:" << endl;
        std::cout << "\t- number of GPUs: " << num_gpus << endl;
        std::cout << "\t- batch size: " << batch_size << endl;
        std::cout << "\t- learning rate: " << learning_rate << endl;
        std::cout << "\t- weight decay: " << weight_decay << endl;
        std::cout << "\t- momentum: " << momentum << endl;        
        std::cout << "\t- patience: " << patience << endl;
        std::cout << "\t- freezing level: " << freezing_level << endl;
        std::cout << "\t- folders: " << root_folders << endl << endl;

        // Start the fine-tuning process
        metric_training(root_folders, model_filename, use_sync_file, learning_rate, batch_size, weight_decay, momentum, patience, freezing_level, gpus);
    } else if (parser.option("lfw-benchmark")) {
        string folders = get_option(parser, "lfw-benchmark", fs::current_path().string());
        const string model_filename = get_option(parser, "model-filename", default_model_filename);
        const long jittering = get_option(parser, "jittering", 10);
        const double threshold = get_option(parser, "threshold", 0.6);
     
        // Load the "standard" DLIB model for face recognition
        model::feats_ssl f_ssl;
        model::feats_std f_ftn;
        const bool use_ssl_model = (model_filename.find(training_filename) != std::string::npos);
        if (use_ssl_model) {
            model::train_ssl anet;
            if (fs::exists(model_filename)) deserialize(model_filename) >> anet;
            else {
                cerr << "Model file <" << model_filename << " not found in the current/given folder" << endl;
                return EXIT_SUCCESS;
            }
            f_ssl.subnet() = anet.subnet();
        } else {
            if (fs::exists(model_filename)) deserialize(model_filename) >> f_ftn;
            else {
                cerr << "Model file <" << model_filename << " not found in the current/given folder" << endl;
                return EXIT_SUCCESS;
            }
        }
        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor sp;
        string shape_predictor_filename = boost::filesystem::current_path().string() + "/shape_predictor_68_face_landmarks_GTX.dat";
        if (fs::exists(shape_predictor_filename)) deserialize(shape_predictor_filename) >> sp;
        else {
            cerr << "Shape predictor file <" << shape_predictor_filename << " not found in the current folder" << endl;
            return EXIT_SUCCESS;
        }

        // To compute stats and efficiency of the model
        running_stats<double> rs, rs_pos, rs_neg;
        std::vector<double> pos_vals, neg_vals;
        std::vector<running_stats<double>> rs_folds(10);
        dlib::rand rnd;
        int cnt = 0;

        // Load images to start the benchmark
        std::cout << std::endl;
        progress_bar* progress = new progress_bar{ std::clog, "LFW benchmark" };
        std::vector<lfw_pair> my_pairs = get_lfw_pairs(folders);
        uint32_t count = 0;
        running_stats<double> rs_dist_pos, rs_dist_neg;
        for (auto& p : my_pairs) {
            progress->write(count++, my_pairs.size());
            if (g_interrupted) break;
            matrix<rgb_pixel> img, crop;
            matrix<float, 0, 1> v1, v2;
            double dist = 0.0;
            try { load_image(img, p.filename1); }
            catch (...) {
                std::cerr << "Error during image loading: " << p.filename1 << std::endl;
                break;
            }
            crop = get_lfw_face_chip(img, p.face1, detector, sp);
            if (use_ssl_model) v1 = mean(mat(f_ssl(quick_jitter_image(crop, jittering))));
            else v1 = mean(mat(f_ftn(quick_jitter_image(crop, jittering))));

            try { load_image(img, p.filename2); }
            catch (...) {
                cerr << "Error during image loading: " << p.filename2 << endl;
                break;
            }
            crop = get_lfw_face_chip(img, p.face2, detector, sp);
            if (use_ssl_model) v2 = mean(mat(f_ssl(quick_jitter_image(crop, jittering))));
            else v2 = mean(mat(f_ftn(quick_jitter_image(crop, jittering))));

            dist = length(v1 - v2);
            if (p.are_same_person) {
                pos_vals.push_back(dist);
                rs_dist_pos.add(dist);
                if (dist < threshold) {
                    rs.add(1);
                    rs_pos.add(1);
                    rs_folds[(cnt / 300) % 10].add(1);
                } else {
                    rs.add(0);
                    rs_pos.add(0);
                    rs_folds[(cnt / 300) % 10].add(0);
                }
            } else {
                neg_vals.push_back(dist);
                rs_dist_neg.add(dist);
                if (dist >= threshold) {
                    rs.add(1);
                    rs_neg.add(1);
                    rs_folds[(cnt / 300) % 10].add(1);
                } else {
                    rs.add(0);
                    rs_neg.add(0);
                    rs_folds[(cnt / 300) % 10].add(0);
                }
            }
            ++cnt;
        }        
        std::cout << std::endl << "overall LFW accuracy: " << std::fixed << std::setprecision(2) << (rs.mean() * 100.0) << "%" << std::endl;
        std::cout << " mean dist_pos: " << rs_dist_pos.mean() << endl;
        std::cout << " mean dist_neg: " << rs_dist_neg.mean() << endl;
        std::cout << " theorical distance threshold: " << ((rs_dist_pos.mean() + rs_dist_neg.mean()) / 2.0) << endl;
        std::cout << " pos LFW accuracy: " << std::fixed << std::setprecision(2) << (rs_pos.mean() * 100.0) << "%" << std::endl;
        std::cout << " neg LFW accuracy: " << std::fixed << std::setprecision(2) << (rs_neg.mean() * 100.0) << "%" << std::endl;
        running_stats<double> rscv;
        for (auto& r : rs_folds) {
            cout << "   fold mean: " << std::fixed << std::setprecision(4) << r.mean() << endl;
            rscv.add(r.mean());
        }
        std::cout << " rscv.mean(): " << std::fixed << std::setprecision(6) << rscv.mean() << std::endl;
        std::cout << " rscv.stddev(): " << std::fixed << std::setprecision(2) << (rscv.stddev() * 1000.0) << "e-3" << std::endl;
        auto err = equal_error_rate(pos_vals, neg_vals);
        std::cout << " ERR accuracy: " << std::fixed << std::setprecision(2) << ((1 - err.first) * 100.0) << "%" << std::endl;
        std::cout << " ERR threshold: " << std::fixed << std::setprecision(2) << (err.second * 100.0) << "%" << std::setprecision(0) << std::endl;
    }
    return EXIT_SUCCESS;
}
catch (const exception& e)
{
    cout << e.what() << endl;
    return EXIT_FAILURE;
}