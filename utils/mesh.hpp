#include <vector>
#include <array>
#include <string>
#include <cassert>
#include <fstream>

/**
 * @brief This class represents a 3D mesh consisting of vertices, vertex colour
 * information and texture coordinates.
 *
 * Additionally it stores the indices that specify which vertices
 * to use to generate the triangle mesh out of the vertices.
 */
struct Mesh
{
	std::vector<cv::Vec4f> vertices; ///< 3D vertex positions.
	std::vector<cv::Vec3f> colors; ///< Colour information for each vertex. Expected to be in RGB order.
	std::vector<cv::Vec2f> texcoords; ///< Texture coordinates for each vertex.

	std::vector<std::array<int, 3>> tvi; ///< Triangle vertex indices
	std::vector<std::array<int, 3>> tci; ///< Triangle color indices
};

/**
 * @brief Writes the given Mesh to an obj file that for example can be read by MeshLab.
 *
 * If the mesh contains vertex colour information, it will be written to the obj as well.
 *
 * @param[in] mesh The mesh to save as obj.
 * @param[in] filename Output filename (including ".obj").
 */
inline void write_obj(Mesh mesh, std::string filename)
{
	assert(mesh.vertices.size() == mesh.colors.size() || mesh.colors.empty());

	std::ofstream obj_file(filename);

	if (mesh.colors.empty()) {
		for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
			obj_file << "v " << mesh.vertices[i][0] << " " << mesh.vertices[i][1] << " " << mesh.vertices[i][2] << " " << std::endl;
		}
	}
	else {
		for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
			obj_file << "v " << mesh.vertices[i][0] << " " << mesh.vertices[i][1] << " " << mesh.vertices[i][2] << " " << mesh.colors[i][0] << " " << mesh.colors[i][1] << " " << mesh.colors[i][2] << " " << std::endl;
		}
	}

	for (auto&& v : mesh.tvi) {
		// Add one because obj starts counting triangle indices at 1
		obj_file << "f " << v[0] << " " << v[1] << " " << v[2] << std::endl;
	}

	return;
}

/**
 * Internal helper function that creates a Mesh from given shape and colour
 * PCA instances. Needs the vertex index lists as well to assemble the mesh -
 * and optional texture coordinates.
 *
 * If \c color is empty, it will create a mesh without vertex colouring.
 *
 * @param[in] shape PCA shape model instance.
 * @param[in] color PCA color model instance.
 * @param[in] tvi Triangle vertex indices.
 * @param[in] tci Triangle color indices (usually identical to the vertex indices).
 * @param[in] texture_coordinates Optional texture coordinates for each vertex.
 * @return A mesh created from given parameters.
 */
Mesh sample_to_mesh(cv::Mat shape, cv::Mat color, std::vector<std::array<int, 3>> tvi, 
	std::vector<std::array<int, 3>> tci, std::vector<cv::Vec2f> texture_coordinates  = std::vector<cv::Vec2f>() )
{
	assert(shape.rows == color.rows || color.empty()); // The number of vertices (= model.getDataDimension() / 3) has to be equal for both models, or, alternatively, it has to be a shape-only model.

	auto num_vertices = shape.rows / 3;

	Mesh mesh;

	// Construct the mesh vertices:
	mesh.vertices.resize(num_vertices);
	for (auto i = 0; i < num_vertices; ++i) {
		mesh.vertices[i] = cv::Vec4f(shape.at<float>(i * 3 + 0), shape.at<float>(i * 3 + 1), shape.at<float>(i * 3 + 2), 1.0f);
	}

	// Assign the vertex color information if it's not a shape-only model:
	if (!color.empty()) {
		mesh.colors.resize(num_vertices);
		for (auto i = 0; i < num_vertices; ++i) {
			mesh.colors[i] = cv::Vec3f(color.at<float>(i * 3 + 0), color.at<float>(i * 3 + 1), color.at<float>(i * 3 + 2));        // order in hdf5: RGB. Order in OCV: BGR. But order in vertex.color: RGB
		}
	}

	// Assign the triangle lists:
	mesh.tvi = tvi;
	mesh.tci = tci; // tci will be empty in case of a shape-only model

	// Texture coordinates, if the model has them:
	if (!texture_coordinates.empty()) {
		mesh.texcoords.resize(num_vertices);
		for (auto i = 0; i < num_vertices; ++i) {
			mesh.texcoords[i] = texture_coordinates[i];
		}
	}

	return mesh;
};

/**
 * Update the vertices coordinate
 */
void update_mesh(cv::Mat shape, Mesh &mesh)
{
	auto num_vertices = shape.rows / 3;
	for (auto i = 0; i < num_vertices; ++i) {
		mesh.vertices[i] = cv::Vec4f(shape.at<float>(i * 3 + 0), shape.at<float>(i * 3 + 1), shape.at<float>(i * 3 + 2), 1.0f);
	}
};