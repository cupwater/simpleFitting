#include <opencv2/core/core.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include "rcr/model.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "helpers.hpp"
#include "matparse.hpp"
#include "mesh.hpp"

#include "fitting/nonlinear_camera_estimation.hpp"


#define SHAPE_BASIS_NUM 199
#define BLEND_BASIS_NUM 80
#define LANDMARKS_NUM   68 

using namespace cv;
using namespace std;

//read the 3D landmarks shape and blendshape basis from file
void readData(vector<Vec3f> &model_points, cv::Mat &shape_basis_3Dlandmarks, cv::Mat &blendshape_basis_3Dlandmarks, string filename)
{
    std::ifstream shape_blend_basis_file(filename);
			
    //write the landmarks in mean shape to file
    for(int i=0; i<model_points.size(); i++)
        for(int j=0; j<3; j++)
            shape_blend_basis_file>>model_points[i][j];

    //write the landmarks in 3D shape to file
    for(int i=0; i<blendshape_basis_3Dlandmarks.rows; i++)
        for(int j=0; j<blendshape_basis_3Dlandmarks.cols; j++)
            shape_blend_basis_file>>blendshape_basis_3Dlandmarks.at<float>(i,j);
    //write the landmarks in 3D blendshape to file
    for(int i=0; i<shape_basis_3Dlandmarks.rows; i++)
        for(int j=0; j<shape_basis_3Dlandmarks.cols; j++)
            shape_blend_basis_file>>shape_basis_3Dlandmarks.at<float>(i,j);
    shape_blend_basis_file.close();
}


//the image points and model points is ok, and the shape_basis_3Dlandmarks blendshape_basis_3Dlandmarks is ok.
//the affine_cam is ok, img is ok, eigenvalues is ok.. we need to compute the shape_coeffs and blend_coeffs.
void myFitting2(vector<Vec2f> image_points, vector<Vec3f> &model_points, cv::Mat shape_basis_3Dlandmarks, 
                cv::Mat blendshape_basis_3Dlandmarks, cv::Mat &shape_coeffs, 
                cv::Mat &blendshape_coeffs, cv::Mat img, cv::Mat eigenvalues)
{
	Mat I = Mat::zeros(SHAPE_BASIS_NUM, SHAPE_BASIS_NUM, CV_32FC1);
	for (int i = 0; i < SHAPE_BASIS_NUM; ++i) {
		I.at<float>(i, i) = 1.0f ;// (eigenvalues.at<float>(i, 0) * eigenvalues.at<float>(i, 0)); // the higher the sigma_squared_2D, the smaller the diagonal entries of Sigma will be
	}

	// The landmarks in matrix notation (in homogeneous coordinates), $3N\times 1$
	Mat y = Mat::ones(2 * LANDMARKS_NUM, 1, CV_32FC1);
	for (int i = 0; i < LANDMARKS_NUM; ++i) {
		y.at<float>(2 * i, 0) = image_points[i][0];
		y.at<float>((2 * i) + 1, 0) = image_points[i][1];
	}


    Mat model_points_mat = Mat::ones(1, 3*LANDMARKS_NUM, CV_32FC1);
    for (int i = 0; i < LANDMARKS_NUM; ++i) {
        model_points_mat.at<float>(0, 3*i  ) = model_points[i][0];
        model_points_mat.at<float>(0, 3*i+1) = model_points[i][1];
        model_points_mat.at<float>(0, 3*i+2) = model_points[i][2];
    }


    cv::Mat current_shape_coeffs = shape_coeffs;
    cv::Mat current_blend_coeffs = blendshape_coeffs;
    cv::Mat affine_cam, affine_23;
	int iterator = 0;
	do // run at least once:
	{
       // mean_plus_shape_and_blend = mean_plus_shape_and_blend + current_blend_coeffs.t() * blendshape_basis_3Dlandmarks;
	   	cv::Mat mean_plus_shape_and_blend = model_points_mat + current_shape_coeffs.t() * shape_basis_3Dlandmarks 
                                                + current_blend_coeffs.t() * blendshape_basis_3Dlandmarks;
        for(int i=0; i<model_points.size(); i++)
        {
            for(int j=0; j<3; j++)
                model_points[i][j] = mean_plus_shape_and_blend.at<float>(0, 3*i+j);
        }
        //get the camera parameters, and using it to get the affine matrix affine_cam
		OrthographicRenderingParameters camera_params = estimate_orthographic_camera(image_points, model_points, img.cols, img.rows);
        affine_cam = get_3x4_affine_camera_matrix(camera_params, img.cols, img.rows);

		//get the right value of transform matrix, parallel transform. matrix T. 
		cv::Mat transformMat = Mat::zeros(LANDMARKS_NUM * 2, 1, CV_32FC1);
		for(int i=0; i<LANDMARKS_NUM; i++)
		{
			transformMat.at<float>(i*2, 0) = affine_cam.at<float>(0,3);
			transformMat.at<float>(i*2+1, 0) = affine_cam.at<float>(1,3);
		}

		for(int i=0; i<3; i++)
		{
			for(int j=0; j<4; j++)
			{
				std::cout<<affine_cam.at<float>(i, j)<<'\t';
			}
			std::cout<<std::endl;
		}
			
				

		affine_23 = Mat::zeros(2, 3, CV_32FC1);
		for(int i=0; i<2; i++)
			for(int j=0; j<3; j++)
				affine_23.at<float>(i, j) = affine_cam.at<float>(i,j);

        //fitting the blendshape
		cv::Mat blendshape_basis_3Dlandmarks_reshape = blendshape_basis_3Dlandmarks.reshape(0, LANDMARKS_NUM * BLEND_BASIS_NUM);
        cv::Mat A_blendshape_reshape = blendshape_basis_3Dlandmarks_reshape * affine_23.t();
		cv::Mat temp_blendshape = A_blendshape_reshape.reshape(0, BLEND_BASIS_NUM);
		cv::Mat A_blendshape = temp_blendshape.t();

		cv::Mat mean_plus_shape =model_points_mat + current_shape_coeffs.t() * shape_basis_3Dlandmarks;
		cv::Mat mean_plus_shape_reshape = mean_plus_shape.reshape(0, LANDMARKS_NUM);
		cv::Mat mean_plus_shape_project = mean_plus_shape_reshape * affine_23.t();
		cv::Mat mean_plus_shape_1 = mean_plus_shape_project.reshape(0, LANDMARKS_NUM * 2);
		
        cv::Mat b_blendshape = y - mean_plus_shape_1 - transformMat;

        cv::Mat AtAReg_blendshape = A_blendshape.t() * A_blendshape + 1000.0f * Mat::eye(BLEND_BASIS_NUM, BLEND_BASIS_NUM, CV_32FC1);
        cv::solve(AtAReg_blendshape, A_blendshape.t() * b_blendshape, current_blend_coeffs, cv::DECOMP_SVD);

        //fitting the shape
		cv::Mat shape_basis_3Dlandmarks_reshape = shape_basis_3Dlandmarks.reshape(0, LANDMARKS_NUM * SHAPE_BASIS_NUM);
        cv::Mat A_shape_reshape = shape_basis_3Dlandmarks_reshape * affine_23.t();
		cv::Mat temp_shape = A_shape_reshape.reshape(0, SHAPE_BASIS_NUM);
		cv::Mat A_shape = temp_shape.t();

		cv::Mat mean_plus_blendshape = model_points_mat + current_blend_coeffs.t() * blendshape_basis_3Dlandmarks;
		cv::Mat mean_plus_blendshape_reshape = mean_plus_blendshape.reshape(0, LANDMARKS_NUM);
		cv::Mat mean_plus_blendshape_project = mean_plus_blendshape_reshape * affine_23.t();
		cv::Mat mean_plus_blendshape_1 = mean_plus_blendshape_project.reshape(0, LANDMARKS_NUM * 2);		

        cv::Mat b_shape = y - mean_plus_blendshape_1 - transformMat;

        cv::Mat AtAReg_shape = A_shape.t() * A_shape + 6.9 * I;
        cv::solve(AtAReg_shape, A_shape.t() * b_shape, current_shape_coeffs, cv::DECOMP_SVD);
		iterator++;
	}while (iterator<3);

	cv::Mat transformMatrix(2, 1, CV_32FC1);
	transformMatrix.at<float>(0, 0) = affine_cam.at<float>(0,3);
	transformMatrix.at<float>(1, 0) = affine_cam.at<float>(1,3);
	

	//the correspondence is correct
	for (int kk = 0; kk < model_points.size(); ++kk ){
		
		cv::Mat pt2d = affine_23 * cv::Mat(model_points[kk]) + transformMatrix;

		cv::rectangle(img, cv::Point2f(pt2d.at<float>(0,0)-1.0f, pt2d.at<float>(1,0)-1.0f), 
			cv::Point2f(pt2d.at<float>(0,0),pt2d.at<float>(1,0)), { 255, 255, 0 });
		
		
		cv::rectangle(img, cv::Point2f(image_points[kk][0] - 1.0f, image_points[kk][1] - 1.0f), 
			cv::Point2f(image_points[kk][0], image_points[kk][1]), { 0, 255, 255 });
	}
	
	cv::imshow("xxxx", img);
}

/**
 * This app demonstrates facial landmark tracking, estimation of the 3D pose
 * and fitting of the shape model of a 3D Morphable Model from a video stream,
 * and merging of the face texture.
 */
int main(int argc, char *argv[])
{

    //first read the detector model and landmarks model
    rcr::detection_model rcr_model;
	// Load the landmark detection model:
	try {
		rcr_model = rcr::load_detection_model("./share/face_landmarks_model_rcr_68.bin");
	}
	catch (const cereal::Exception& e) {
		cout << "Error reading the RCR model " << "landmarkdetector" << ": " << e.what() << endl;
		return EXIT_FAILURE;
	}

	// Load the face detector from OpenCV:
	cv::CascadeClassifier face_cascade;
	if (!face_cascade.load("./share/haarcascade_frontalface_alt2.xml"))
	{
		cout << "Error loading the face detector " << "facedetector" << "." << endl;
		return EXIT_FAILURE;
	}

	cv::VideoCapture cap;
	/*if (!cap.isOpened()) {
		cout << "Couldn't open the given file or camera 0." << endl;
		return EXIT_FAILURE;
	}*/
	
	cv::namedWindow("video", 1);
	cv::namedWindow("xxxx", 1);

	Mat frame, unmodified_frame;

	bool have_face = false;
	rcr::LandmarkCollection<Vec2f> current_landmarks;

    // Fit the 3DMM. First, estimate the pose:
    vector<Vec2f> image_points(LANDMARKS_NUM); // the 2D landmark points for which a mapping to the 3D model exists
    vector<Vec3f> model_points(LANDMARKS_NUM); // the corresponding points in the 3D shape model
	 // the correspondence points in all the 3D shape basis
    cv::Mat shape_basis_3Dlandmarks = Mat::zeros(SHAPE_BASIS_NUM, 3 * LANDMARKS_NUM, CV_32FC1);
    // the correspondence points in all the 3D blendshape basis
    cv::Mat blendshape_basis_3Dlandmarks = Mat::zeros(BLEND_BASIS_NUM, 3 * LANDMARKS_NUM, CV_32FC1);

	cv::Mat blendshapeMat, shapeMat, mean_shape, mean_color;
	std::vector<std::array<int, 3>> triangle_list;

	//read the basis data from model.mat file
	read3DbasisData(mean_shape, mean_color, shapeMat, blendshapeMat, triangle_list, "./share/Model.mat");

    readData(model_points, shape_basis_3Dlandmarks, blendshape_basis_3Dlandmarks, "./share/basis2landmarks.txt");
	
	//cv::Mat 

    for(int i=0; i<model_points.size(); i++)
        for(int j=0; j<3; j++)
            model_points[i][j] *= 1e2;
	
	shape_basis_3Dlandmarks *= 1e2;
	blendshape_basis_3Dlandmarks *= 1e2;

    vector<int> vectex_ids; // their vertex indices

	for (;;)
	{
		//cap >> frame; // get a new frame from camera
		frame = cv::imread("./share/test.png");
		if (frame.empty()) { // stop if we're at the end of the video
			break;
		}
		
		// We do a quick check if the current face's width is <= 50 pixel. If it is, 
        //we re-initialise the tracking with the face detector.
		if (have_face && get_enclosing_bbox(rcr::to_row(current_landmarks)).width <= 50) {
			std::cout << "Reinitialising because the face bounding-box width is <= 50 px" << std::endl;
			have_face = false;
		}

		unmodified_frame = frame.clone();

		if (!have_face) {
			// Run the face detector and obtain the initial estimate using the mean landmarks:
			vector<Rect> detected_faces;
			face_cascade.detectMultiScale(unmodified_frame, detected_faces, 1.2, 2, 0, cv::Size(110, 110));
			if (detected_faces.empty()) {
				cv::imshow("video", frame);
				cv::waitKey(30);
				continue;
			}
			cv::rectangle(frame, detected_faces[0], { 255, 0, 0 });
			Rect ibug_facebox = rescale_facebox(detected_faces[0], 0.85, 0.2);

			current_landmarks = rcr_model.detect(unmodified_frame, ibug_facebox);
			rcr::draw_landmarks(unmodified_frame, current_landmarks);

			have_face = true;
		}
		else {
			auto enclosing_bbox = get_enclosing_bbox(rcr::to_row(current_landmarks));
			enclosing_bbox = make_bbox_square(enclosing_bbox);
			current_landmarks = rcr_model.detect(unmodified_frame, enclosing_bbox);
			rcr::draw_landmarks(frame, current_landmarks, { 0, 255, 0 }); // green, the new optimised landmarks
		}
		// Fit the PCA shape model and expression blendshapes:
		cv::Mat shape_coefficients = Mat::zeros(199, 1, CV_32FC1);
		cv::Mat blendshape_coefficients = Mat::zeros(80, 1, CV_32FC1);
        cv::Mat eigenvalues;
        image_points.clear();
        // Sub-select all the landmarks which we have a mapping for (i.e. that are defined in the 3DMM):
        for (int i = 0; i < current_landmarks.size(); ++i) {
            image_points.emplace_back(current_landmarks[i].coordinates);
        }

		myFitting2(image_points, model_points, shape_basis_3Dlandmarks, blendshape_basis_3Dlandmarks,
               shape_coefficients, blendshape_coefficients, unmodified_frame, eigenvalues);
		
		cv::Mat finalShape = mean_shape + shapeMat*shape_coefficients + blendshapeMat*blendshape_coefficients;

		//Get the fitted mesh, extract the texture:
		Mesh mesh = sample_to_mesh(finalShape, mean_color, triangle_list, triangle_list);
		
		write_obj(mesh, "shape.obj");

		cv::waitKey(0);

	}
	return EXIT_SUCCESS;
};