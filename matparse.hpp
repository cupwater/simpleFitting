#pragma once

#include <string>
#include <fstream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "MatlabIO.hpp"
#include "MatlabIOContainer.hpp"

using namespace cv;
using namespace std;

using cv::Mat;
using cv::Vec2f;
using cv::Vec3f;
using cv::Vec4f;
using std::cout;
using std::endl;
using std::vector;

/**
 * this app parse the .mat file 
 */
bool read3DbasisData(cv::Mat &mean_shape, cv::Mat &mean_color, cv::Mat &shapeMat, cv::Mat &blendshapesMat, 
    std::vector<std::array<int, 3>> &triangle_list, string filename)
{
    // create a new reader
    MatlabIO matio;
    bool ok = matio.open(filename, "r");
    if (!ok)
        return false;

    // read all of the variables in the file
    vector<MatlabIOContainer> variables;
    variables = matio.read();

    // close the file
    matio.close();
    // display the file info
    matio.whos(variables);

    for (unsigned int n = 0; n < variables.size(); ++n)
    {
        //get the shape information
        if (variables[n].name().compare("mu_shape") == 0)
        {
            mean_shape = variables[n].data<Mat>();
            std::cout<<mean_shape.rows<<'\t'<<mean_shape.cols<<std::endl;
        }
        //get the shape information
        if (variables[n].name().compare("mu_color") == 0)
        {
            mean_color = variables[n].data<Mat>();
            std::cout<<mean_color.rows<<'\t'<<mean_color.cols<<std::endl;
        }
        if (variables[n].name().compare("pca_shape") == 0)
        {
            shapeMat = variables[n].data<Mat>();
            std::cout<<shapeMat.rows<<'\t'<<shapeMat.cols<<std::endl;
        }
        if (variables[n].name().compare("expr_disp_intel") == 0)
        {
            blendshapesMat =  variables[n].data<Mat>();
            std::cout<<blendshapesMat.rows<<'\t'<<blendshapesMat.cols<<std::endl;
        }
         if (variables[n].name().compare("tri") == 0)
        {
            cv::Mat tri = variables[n].data<Mat>();
            std::cout<<tri.rows<<'\t'<<tri.cols<<std::endl;
            for (int r = 0; r < tri.rows; r++)
            {
                std::array<int, 3> temp;
                temp[0] = (int)(tri.at<float>(r,0));
                temp[1] = (int) (tri.at<float>(r,1));
                temp[2] = (int) (tri.at<float>(r,2));
                triangle_list.push_back(temp);
            }

        }         
    }
    return true;
}
