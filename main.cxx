/* The following code is mostly as-is as it was from the Dlib website's example
 * on how to implement Deep Face Recognition. It was modified to work with Rust
 * interopability. The gui code and most comments were removed, and transformed
 * into a library instead of an executable in its own right.
 *
 * See here for original code: http://dlib.net/dnn_face_recognition_ex.cpp.html */

#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>

using namespace dlib;
using namespace std;

template <template <int, template <typename> class, int, typename> class block, int N, template <typename> class BN, typename SUBNET>
using residual = add_prev1<block<N, BN, 1, tag1<SUBNET>>>;

template <template <int, template <typename> class, int, typename> class block, int N, template <typename> class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<block<N, BN, 2, tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<con<N, 3, 3, 1, 1, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET>
using ares = relu<residual<block, N, affine, SUBNET>>;
template <int N, typename SUBNET>
using ares_down = relu<residual_down<block, N, affine, SUBNET>>;

template <typename SUBNET>
using alevel0 = ares_down<256, SUBNET>;
template <typename SUBNET>
using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template <typename SUBNET>
using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template <typename SUBNET>
using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template <typename SUBNET>
using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128, avg_pool_everything<alevel0<alevel1<alevel2<alevel3<alevel4<max_pool<3, 3, 2, 2, relu<affine<con<32, 7, 7, 2, 2, input_rgb_image_sized<150>>>>>>>>>>>>>;

class Guise
{
public:
    Guise();
    ~Guise(){};

    std::map<rectangle, rectangle> compare_images(string file_one, string file_two);
    bool compare_face_rectangles(std::pair<rectangle, rectangle> pair, string file_one, string file_two);

private:
    bool compare_face_rectangles(std::pair<rectangle, rectangle> pair, matrix<rgb_pixel> img_one, matrix<rgb_pixel> img_two);
    std::vector<rectangle> get_faces(matrix<rgb_pixel> img);
    bool compare_faces(matrix<rgb_pixel> &face_one, matrix<rgb_pixel> &face_two);

    // AI.
    frontal_face_detector detector;
    shape_predictor sp;
    anet_type net;
};

Guise::Guise()
{
    detector = get_frontal_face_detector();
    deserialize("./shape_predictor_5_face_landmarks.dat") >> sp;
    deserialize("./dlib_face_recognition_resnet_model_v1.dat") >> net;
}

// Get the face rectangle and the shape detection.
std::vector<rectangle> Guise::get_faces(matrix<rgb_pixel> img)
{
    return detector(img);
}

// Get retangle coordinates of faces that are the same.
std::map<rectangle, rectangle> Guise::compare_images(string file_one, string file_two)
{
    // The first retangle references file_one, the second references file_two.
    std::map<rectangle, rectangle> map;
    // Vector of faces from the first image.
    std::vector<matrix<rgb_pixel>> faces;
    // Vector of faces from the second image.
    std::vector<matrix<rgb_pixel>> faces2;
    // Image storage.
    matrix<rgb_pixel> img;
    matrix<rgb_pixel> img2;
    // Temporary retangle data.
    std::vector<rectangle> tmp;

    // Extract all faces from the first image.
    load_image(img, file_one);
    tmp = get_faces(img);

    // Extract all faces from the second image and compare.
    load_image(img2, file_two);
    for (auto face : get_faces(img2))
    {
        for (int i = 0; i < tmp.size(); i++)
        {
            std::pair<rectangle, rectangle> pair;
            pair.first = tmp[i];
            pair.second = face;
            bool res = compare_face_rectangles(pair, img, img2);

            if (res == true)
            {
                map[pair.first] = pair.second;
                cout << "Found pair!" << endl;
            }
        }
    }
    return map;
}

// Compare two rectangles.
bool Guise::compare_face_rectangles(std::pair<rectangle, rectangle> pair, string file_one, string file_two)
{
    matrix<rgb_pixel> img;
    load_image(img, file_one);

    matrix<rgb_pixel> img_two;
    load_image(img_two, file_two);

    return compare_face_rectangles(pair, img, img_two);
}

// Compare two rectangles. Please pass the whole image.
bool Guise::compare_face_rectangles(std::pair<rectangle, rectangle> pair, matrix<rgb_pixel> img_one, matrix<rgb_pixel> img_two)
{
    // Extract first face chip using rectangle and image.
    auto val = sp(img_one, pair.first);
    matrix<rgb_pixel> face_chip;
    extract_image_chip(img_one, get_face_chip_details(val, 150, 0.25), face_chip);
    face_chip = move(face_chip);

    val = sp(img_two, pair.second);
    matrix<rgb_pixel> face_chip2;
    extract_image_chip(img_two, get_face_chip_details(val, 150, 0.25), face_chip2);
    face_chip2 = move(face_chip2);

    return compare_faces(face_chip, face_chip2);
}

bool Guise::compare_faces(matrix<rgb_pixel> &face_one, matrix<rgb_pixel> &face_two)
{
    // Convert each face image in faces into a 128D vector.
    std::vector<matrix<float, 0, 1>> face_descriptors;
    face_descriptors.push_back(net(face_one));
    face_descriptors.push_back(net(face_two));

    // Graph the two faces to see if they are similar enough.
    std::vector<sample_pair> edges;
    for (size_t i = 0; i < face_descriptors.size(); ++i)
    {
        for (size_t j = i; j < face_descriptors.size(); ++j)
        {
            // If the similarity of the images is close enough, then mark.
            auto len = length(face_descriptors[i] - face_descriptors[j]);
            if (len < .6)
                edges.push_back(sample_pair(i, j));
        }
    }

    // Number of individials may be less and the number of actual faces in the image.
    std::vector<unsigned long> people; // An array of the two faces identified with an id (a number starting at zero)
    const auto number_of_individuals = chinese_whispers(edges, people);

    if (number_of_individuals == 1)
    {
        return true;
    }
    else
    {
        return false;
    }
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        return 1;
    }

    Guise guise;
    guise.compare_images(string(argv[1]), string(argv[2]));
}