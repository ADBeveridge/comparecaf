#pragma once

#include <dlib/dnn.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>


using namespace dlib;
template <template <int, template <typename> class, int, typename> class block, int N, template <typename> class BN, typename SUBNET>
using residual = add_prev1<block<N, BN, 1, tag1<SUBNET>>>;

template <template <int, template <typename> class, int, typename> class block, int N, template <typename> class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<block<N, BN, 2, tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<dlib::con<N, 3, 3, 1, 1, dlib::relu<BN<dlib::con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET>
using ares = dlib::relu<residual<block, N, affine, SUBNET>>;
template <int N, typename SUBNET>
using ares_down = dlib::relu<residual_down<block, N, affine, SUBNET>>;

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

namespace guise
{
    class Guise
    {
    public:
        Guise();
        ~Guise(){};

        std::map<dlib::rectangle, dlib::rectangle> compare_images(std::string file_one, std::string file_two);
        bool compare_face_rectangles(std::pair<dlib::rectangle, dlib::rectangle> pair, std::string file_one, std::string file_two);

    private:
        bool compare_face_rectangles(std::pair<dlib::rectangle, dlib::rectangle> pair, dlib::matrix<dlib::rgb_pixel> img_one, dlib::matrix<dlib::rgb_pixel> img_two);
        std::vector<dlib::rectangle> get_faces(dlib::matrix<dlib::rgb_pixel> img);
        bool compare_faces(dlib::matrix<dlib::rgb_pixel> &face_one, dlib::matrix<dlib::rgb_pixel> &face_two);

        // AI.
        dlib::frontal_face_detector detector;
        dlib::shape_predictor sp;
        anet_type net;
    };
}
