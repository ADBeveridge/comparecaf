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

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET> 
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<alevel0<alevel1<alevel2<alevel3<alevel4<max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,input_rgb_image_sized<150>>>>>>>>>>>>>;

std::vector<matrix<rgb_pixel>> jitter_image(const matrix<rgb_pixel>& img);

int main(int argc, char** argv) try
{
    if (argc != 3)
    {
        return 1;
    }

    // Load CNN.
    frontal_face_detector detector = get_frontal_face_detector();
    shape_predictor shape_predictor;
    deserialize("./shape_predictor_5_face_landmarks.dat") >> shape_predictor;
    anet_type net;
    deserialize("./dlib_face_recognition_resnet_model_v1.dat") >> net;

    matrix<rgb_pixel> img;
    load_image(img, argv[1]);

    matrix<rgb_pixel> img2;
    load_image(img, argv[2]);
    


    // Display the raw image on the screen
    image_window win(img); 

    // Extract all faces from the image.
    std::vector<matrix<rgb_pixel>> faces;
    for (auto face : detector(img))
    {
        auto shape = shape_predictor(img, face);
        matrix<rgb_pixel> face_chip;
        extract_image_chip(img, get_face_chip_details(shape,150,0.25), face_chip);
        faces.push_back(move(face_chip));

        win.add_overlay(face);
    }

    if (faces.size() == 0)
    {
        cout << "No faces found in image!" << endl;
        return 1;
    }

    // Convert each face image in faces into a 128D vector.
    std::vector<matrix<float,0,1>> face_descriptors = net(faces);

    // Build a graph of faces that are similar enough.
    std::vector<sample_pair> edges;
    for (size_t i = 0; i < face_descriptors.size(); ++i)
    {
        for (size_t j = i; j < face_descriptors.size(); ++j)
        {
            // If the similarity of the images is close enough, then mark.
            if (length(face_descriptors[i]-face_descriptors[j]) < 0.6)
                edges.push_back(sample_pair(i,j));
        }
    }// Build a graph of faces that are similar enough.
    std::vector<sample_pair> edges;
    for (size_t i = 0; i < face_descriptors.size(); ++i)
    {
        for (size_t j = i; j < face_descriptors.size(); ++j)
        {
            // If the similarity of the images is close enough, then mark.
            if (length(face_descriptors[i]-face_descriptors[j]) < 0.6)
                edges.push_back(sample_pair(i,j));
        }
    }

    std::vector<unsigned long> people; // An array of faces itentified with a number starting at zero.

    // Number of individials may be less and the number of actual faces in the image.
    const auto number_of_individuals = chinese_whispers(edges, people);
    // This will correctly indicate that there are 4 people in the image.
    cout << "number of people found in the image: "<< number_of_individuals << endl;


    // Now let's display the face clustering results on the screen.  You will see that it
    // correctly grouped all the faces. 
    std::vector<image_window> win_clusters(number_of_individuals);
    for (size_t person_id = 0; person_id < number_of_individuals; ++person_id)
    {
        std::vector<matrix<rgb_pixel>> temp;
        for (size_t j = 0; j < people.size(); ++j)
        {
            if (person_id == people[j])
                temp.push_back(faces[j]);
        }
        win_clusters[person_id].set_title("face cluster " + cast_to_string(person_id));
        win_clusters[person_id].set_image(tile_images(temp));
    }

    cout << "hit enter to terminate" << endl;
    cin.get();
}
catch (std::exception& e)
{
    cout << e.what() << endl;
}


std::vector<matrix<rgb_pixel>> jitter_image(
    const matrix<rgb_pixel>& img
)
{
    // All this function does is make 100 copies of img, all slightly jittered by being
    // zoomed, rotated, and translated a little bit differently. They are also randomly
    // mirrored left to right.
    thread_local dlib::rand rnd;

    std::vector<matrix<rgb_pixel>> crops; 
    for (int i = 0; i < 100; ++i)
        crops.push_back(jitter_image(img,rnd));

    return crops;
}
