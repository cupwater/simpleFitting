#pragma once

#ifndef APP_HELPERS_HPP_
#define APP_HELPERS_HPP_

#include "rcr/landmark.hpp"

#include "opencv2/core/core.hpp"

#include <vector>
#include <algorithm>
#include <iterator>
#include <cassert>

/**
 * @brief Scales and translates a facebox. Useful for converting
 * between face boxes from different face detectors.
 *
 * To convert from V&J faceboxes to ibug faceboxes, use a scaling
 * of 0.85 and a translation_y of 0.2.
 * Ideally, we would learn the exact parameters from data.
 *
 * @param[in] facebox Input facebox.
 * @param[in] scaling The input facebox will be scaled by this factor.
 * @param[in] translation_y How much, in percent of the original facebox's width, the facebox will be translated in y direction. A positive value means facebox moves downwards.
 * @return The rescaled facebox.
 */
cv::Rect rescale_facebox(cv::Rect facebox, float scaling, float translation_y)
{
	// Assumes a square input facebox to work? (width==height)
	const auto new_width = facebox.width * scaling;
	const auto smaller_in_px = facebox.width - new_width;
	const auto new_tl = facebox.tl() + cv::Point2i(smaller_in_px / 2.0f, smaller_in_px / 2.0f);
	const auto new_br = facebox.br() - cv::Point2i(smaller_in_px / 2.0f, smaller_in_px / 2.0f);
	cv::Rect rescaled_facebox(new_tl, new_br);
	rescaled_facebox.y += facebox.width * translation_y;
	return rescaled_facebox;
};

/**
 * @brief Calculates the bounding box that encloses the landmarks.
 *
 * The bounding box will not be square.
 *
 * @param[in] landmarks Landmarks.
 * @return The enclosing bounding box.
 */
template<class T = int>
cv::Rect_<T> get_enclosing_bbox(cv::Mat landmarks)
{
	auto num_landmarks = landmarks.cols / 2;
	double min_x_val, max_x_val, min_y_val, max_y_val;
	cv::minMaxLoc(landmarks.colRange(0, num_landmarks), &min_x_val, &max_x_val);
	cv::minMaxLoc(landmarks.colRange(num_landmarks, landmarks.cols), &min_y_val, &max_y_val);
	return cv::Rect_<T>(min_x_val, min_y_val, max_x_val - min_x_val, max_y_val - min_y_val);
};

/**
 * @brief Makes the given face bounding box square by enlarging the
 * smaller of the width or height to be equal to the bigger one.
 *
 * @param[in] bounding_box Input bounding box.
 * @return The bounding box with equal width and height.
 */
cv::Rect make_bbox_square(cv::Rect bounding_box)
{
	auto center_x = bounding_box.x + bounding_box.width / 2.0;
	auto center_y = bounding_box.y + bounding_box.height / 2.0;
	auto box_size = std::max(bounding_box.width, bounding_box.height);
	return cv::Rect(center_x - box_size / 2.0, center_y - box_size / 2.0, box_size, box_size);
};


/**
 * @brief Concatenates two std::vector's of the same type and returns the concatenated
 * vector. The elements of the second vector are appended after the first one.
 *
 * @param[in] vec_a First vector.
 * @param[in] vec_b Second vector.
 * @return The concatenated vector.
 */
template <class T>
auto concat(const std::vector<T>& vec_a, const std::vector<T>& vec_b)
{
	std::vector<T> concatenated_vec;
	concatenated_vec.reserve(vec_a.size() + vec_b.size());
	concatenated_vec.insert(std::end(concatenated_vec), std::begin(vec_a), std::end(vec_a));
	concatenated_vec.insert(std::end(concatenated_vec), std::begin(vec_b), std::end(vec_b));
	return concatenated_vec;
};

#endif /* APP_HELPERS_HPP_ */
