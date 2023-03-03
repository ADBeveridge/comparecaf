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


class Guise {
public:
    Guise();
    ~Guise(){};

    void compare_images(string file_one, string file_two);
private:
    
    bool compare_faces(matrix<rgb_pixel> &face_one, matrix<rgb_pixel> &face_two);
    frontal_face_detector detector;
    shape_predictor sp;
};

Guise::Guise() {
    detector = get_frontal_face_detector();
    deserialize("./shape_predictor_5_face_landmarks.dat") >> sp;
}

void Guise::compare_images(string file_one, string file_two)
{
    // Vector of faces from the first image.
    std::vector<matrix<rgb_pixel>> faces;
    // Vector of faces from the second image.
    std::vector<matrix<rgb_pixel>> faces2;

    // Extract all faces from the first image.
    matrix<rgb_pixel> img;
    load_image(img, file_one);
    for (auto face : detector(img))
    {
        auto shape = sp(img, face);
        matrix<rgb_pixel> face_chip;
        extract_image_chip(img, get_face_chip_details(shape,150,0.25), face_chip);
        faces.push_back(move(face_chip));
    }

    // Extract all faces from the first image.
    matrix<rgb_pixel> img2;
    load_image(img2, file_two);
    for (auto face : detector(img2))
    {
        auto shape = sp(img2, face);
        matrix<rgb_pixel> face_chip;
        extract_image_chip(img, get_face_chip_details(shape,150,0.25), face_chip);
        faces2.push_back(move(face_chip));
    }

    std::cout << "Result: " << compare_faces(faces[0], faces2[0]) << endl;
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

bool Guise::compare_faces(matrix<rgb_pixel> &face_one, matrix<rgb_pixel> &face_two)
{
    anet_type net;
    deserialize("./dlib_face_recognition_resnet_model_v1.dat") >> net;

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
            if (length(face_descriptors[i] - face_descriptors[j]) < 1)
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