#pragma once

#ifndef NONLINEARCAMERAESTIMATION_HPP_
#define NONLINEARCAMERAESTIMATION_HPP_

#include "nonlinear_camera_estimation_detail.hpp"

#include "glm/gtc/matrix_transform.hpp"

#include "Eigen/Geometry"
#include "unsupported/Eigen/NonLinearOptimization"

#include "opencv2/core/core.hpp"

#include <vector>
#include <cassert>


/**
 * @brief A class representing a camera viewing frustum. At the
 * moment used as orthographic camera only.
 */
struct Frustum
{
	float l, r, b, t;
	// optional<float> n, f;
};

/**
 * @brief Represents a set of estimated model parameters (rotation, translation) and
 * camera parameters (viewing frustum).
 *
 * The estimated rotation and translation transform the model from model-space to camera-space,
 * and, if one wishes to use OpenGL, can be used to build the model-view matrix.
 * The parameters are the inverse of the camera position in 3D space.
 *
 * The camera frustum describes the size of the viewing plane of the camera, and
 * can be used to build an OpenGL-conformant orthographic projection matrix.
 *
 * Together, these parameters fully describe the imaging process of a given model instance
 * (under an orthographic projection).
 *
 * The rotation values are given in radians and estimated using the RPY convention.
 * Yaw is applied first to the model, then pitch, then roll (R * P * Y * vertex).
 */
struct OrthographicRenderingParameters
{
	float r_x; // Pitch.
	float r_y; // Yaw. Positive means subject is looking left (we see her right cheek).
	float r_z; // Roll. Positive means the subject's right eye is further down than the other one (he tilts his head to the right).
	float t_x;
	float t_y;
	Frustum frustum;
};

/**
 * @brief Converts a glm::mat4x4 to a cv::Mat.
 *
 * Note: move to render namespace
 */
cv::Mat to_mat(const glm::mat4x4& glm_matrix)
{
	// glm stores its matrices in col-major order in memory, OpenCV in row-major order.
	// Hence we transpose the glm matrix to flip the memory layout, and then point OpenCV
	// to that location.
	auto glm_matrix_t = glm::transpose(glm_matrix);
	cv::Mat opencv_mat(4, 4, CV_32FC1, &glm_matrix_t[0]);
	// we need to clone because the underlying data of the original goes out of scope
	return opencv_mat.clone();
};

/**
 * @brief Creates a 4x4 model-view matrix from given fitting parameters.
 *
 * Together with the Frustum information, this describes the full
 * orthographic rendering parameters of the OpenGL pipeline.
 * Example:
 *
 * @code
 * fitting::OrthographicRenderingParameters rendering_params = ...;
 * glm::mat4x4 view_model = get_4x4_modelview_matrix(rendering_params);
 * glm::mat4x4 ortho_projection = glm::ortho(rendering_params.frustum.l, rendering_params.frustum.r, rendering_params.frustum.b, rendering_params.frustum.t);
 * glm::vec4 viewport(0, image.rows, image.cols, -image.rows); // flips y, origin top-left, like in OpenCV
 *
 * // project a point from 3D to 2D:
 * glm::vec3 point_3d = ...; // from a mesh for example
 * glm::vec3 point_2d = glm::project(point_3d, view_model, ortho_projection, viewport);
 * @endcode
 */
glm::mat4x4 get_4x4_modelview_matrix(OrthographicRenderingParameters params)
{
	// rotation order: RPY * P
	auto rot_mtx_x = glm::rotate(glm::mat4(1.0f), params.r_x, glm::vec3{ 1.0f, 0.0f, 0.0f });
	auto rot_mtx_y = glm::rotate(glm::mat4(1.0f), params.r_y, glm::vec3{ 0.0f, 1.0f, 0.0f });
	auto rot_mtx_z = glm::rotate(glm::mat4(1.0f), params.r_z, glm::vec3{ 0.0f, 0.0f, 1.0f });
	auto t_mtx = glm::translate(glm::mat4(1.0f), glm::vec3{ params.t_x, params.t_y, 0.0f });
	auto modelview = t_mtx * rot_mtx_z * rot_mtx_x * rot_mtx_y;
	return modelview;
};

/**
 * @brief Creates a 3x4 affine camera matrix from given fitting parameters. The
 * matrix transforms points directly from model-space to screen-space.
 *
 * This function is mainly used since the linear shape fitting fitting::fit_shape_to_landmarks_linear
 * expects one of these 3x4 affine camera matrices, as well as render::extract_texture.
 */
cv::Mat get_3x4_affine_camera_matrix(OrthographicRenderingParameters params, int width, int height)
{
	auto view_model = to_mat(get_4x4_modelview_matrix(params));
	auto ortho_projection = to_mat(glm::ortho(params.frustum.l, params.frustum.r, params.frustum.b, params.frustum.t));
	cv::Mat mvp = ortho_projection * view_model;

	glm::vec4 viewport(0, height, width, -height); // flips y, origin top-left, like in OpenCV
	// equivalent to what glm::project's viewport does, but we don't change z and w:
	cv::Mat viewport_mat = (cv::Mat_<float>(4, 4) << viewport[2] / 2.0f, 0.0f,       0.0f, viewport[2] / 2.0f + viewport[0],
												     0.0f,               viewport[3] / 2.0f, 0.0f, viewport[3] / 2.0f + viewport[1],
													 0.0f,               0.0f,               1.0f, 0.0f,
													 0.0f,               0.0f,               0.0f, 1.0f);

	cv::Mat full_projection_4x4 = viewport_mat * mvp;
	cv::Mat full_projection_3x4 = full_projection_4x4.rowRange(0, 3); // we take the first 3 rows, but then set the last one to [0 0 0 1]
	full_projection_3x4.at<float>(2, 0) = 0.0f;
	full_projection_3x4.at<float>(2, 1) = 0.0f;
	full_projection_3x4.at<float>(2, 2) = 0.0f;
	full_projection_3x4.at<float>(2, 3) = 1.0f;

	return full_projection_3x4;
};

/**
 * @brief Returns a glm/OpenGL compatible viewport vector that flips y and
 * has the origin on the top-left, like in OpenCV.
 *
 * Note: Move to detail namespace / not used at the moment.
 */
glm::vec4 get_opencv_viewport(int width, int height)
{
	return glm::vec4(0, height, width, -height);
};

/**
 * @brief This algorithm estimates the rotation angles and translation of the model, as
 * well as the viewing frustum of the camera, given a set of corresponding 2D-3D points.
 *
 * It assumes an orthographic camera and estimates 6 parameters,
 * [r_x, r_y, r_z, t_x, t_y, frustum_scale], where the first five describe how to transform
 * the model, and the last one describes the cameras viewing frustum (see CameraParameters).
 * This 2D-3D correspondence problem is solved using Eigen's LevenbergMarquardt algorithm.
 *
 * The method is slightly inspired by "Computer Vision: Models Learning and Inference",
 * Simon J.D. Prince, 2012, but different in a lot of respects.
 *
 * Eigen's LM implementation requires at least 6 data points, so we require >= 6 corresponding points.
 *
 * Notes/improvements:
 * The algorithm works reliable as it is, however, it could be improved with the following:
 *  - A better initial guess (see e.g. Prince)
 *  - Using the analytic derivatives instead of Eigen::NumericalDiff - they're easy to calculate.
 *
 * @param[in] image_points A list of 2D image points.
 * @param[in] model_points Corresponding points of a 3D model.
 * @param[in] width Width of the image (or viewport).
 * @param[in] height Height of the image (or viewport).
 * @return The estimated model and camera parameters.
 */
OrthographicRenderingParameters estimate_orthographic_camera(std::vector<cv::Vec2f> image_points, std::vector<cv::Vec3f> model_points, int width, int height)
{
	using cv::Mat;
	assert(image_points.size() == model_points.size());
	assert(image_points.size() >= 6); // Number of correspondence points given needs to be equal to or larger than 6

	const float aspect = static_cast<float>(width) / height;

	// Set up the initial parameter vector and the cost function:
	int num_params = 6;
	Eigen::VectorXd parameters; // [rot_x_pitch, rot_y_yaw, rot_z_roll, t_x, t_y, frustum_scale]
	parameters.setConstant(num_params, 0.0); // Set all 6 values to zero (except frustum_scale, see next line)
	parameters[5] = 108.0; // This is just a rough hand-chosen scaling estimate - we could do a lot better. But it works.
	OrthographicParameterProjection cost_function(image_points, model_points, width, height);

	// Note: we have analytical derivatives, so we should use them!
	Eigen::NumericalDiff<OrthographicParameterProjection> cost_function_with_derivative(cost_function, 0.0001);
	// I had to change the default value of epsfcn, it works well around 0.0001. It couldn't produce the derivative with the default, I guess the changes in the gradient were too small.

	Eigen::LevenbergMarquardt<Eigen::NumericalDiff<OrthographicParameterProjection>> lm(cost_function_with_derivative);
	auto info = lm.minimize(parameters); // we could or should use the return value
	// 'parameters' contains the solution now.

	Frustum camera_frustum{ -1.0f * aspect * static_cast<float>(parameters[5]), 1.0f * aspect * static_cast<float>(parameters[5]), 
		-1.0f * static_cast<float>(parameters[5]), 1.0f * static_cast<float>(parameters[5]) };
	return OrthographicRenderingParameters{ static_cast<float>(parameters[0]), static_cast<float>(parameters[1]), 
		static_cast<float>(parameters[2]), static_cast<float>(parameters[3]), static_cast<float>(parameters[4]), camera_frustum };
};


#endif /* NONLINEARCAMERAESTIMATION_HPP_ */
