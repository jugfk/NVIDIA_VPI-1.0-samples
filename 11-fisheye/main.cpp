/*
* Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:
*  * Redistributions of source code must retain the above copyright
*    notice, this list of conditions and the following disclaimer.
*  * Redistributions in binary form must reproduce the above copyright
*    notice, this list of conditions and the following disclaimer in the
*    documentation and/or other materials provided with the distribution.
*  * Neither the name of NVIDIA CORPORATION nor the names of its
*    contributors may be used to endorse or promote products derived
*    from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
* PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
* CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
* EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
* PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
* PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
* OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <opencv2/core/version.hpp>

#if CV_MAJOR_VERSION >= 3
#    include <opencv2/imgcodecs.hpp>
#else
#    include <opencv2/highgui/highgui.hpp>
#endif

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vpi/OpenCVInterop.hpp>

#include <string.h> // for basename(3) that doesn't modify its argument
#include <unistd.h> // for getopt
#include <vpi/Image.h>
#include <vpi/LensDistortionModels.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/Remap.h>

#include <iostream>
#include <sstream>

#define CHECK_STATUS(STMT)                                    \
    do                                                        \
    {                                                         \
        VPIStatus status = (STMT);                            \
        if (status != VPI_SUCCESS)                            \
        {                                                     \
            char buffer[VPI_MAX_STATUS_MESSAGE_LENGTH];       \
            vpiGetLastStatusMessage(buffer, sizeof(buffer));  \
            std::ostringstream ss;                            \
            ss << vpiStatusGetName(status) << ": " << buffer; \
            throw std::runtime_error(ss.str());               \
        }                                                     \
    } while (0);

static void PrintUsage(const char *progname, std::ostream &out)
{
    out << "Usage: " << progname << " <-c W,H> [-s win] <image1> [image2] [image3] ...\n"
        << " where,\n"
        << " W,H\tcheckerboard with WxH squares\n"
        << " win\tsearch window width around checkerboard vertex used\n"
        << "\tin refinement, default is 0 (disable refinement)\n"
        << " imageN\tinput images taken with a fisheye lens camera" << std::endl;
}

struct Params
{
    cv::Size vtxCount;                // Number of internal vertices the checkerboard has
    int searchWinSize;                // search window size around the checkerboard vertex for refinement.
    std::vector<const char *> images; // input image names.
};

static Params ParseParameters(int argc, char *argv[])
{
    Params params = {};

    cv::Size cbSize;

    opterr = 0;
    int opt;
    while ((opt = getopt(argc, argv, "hc:s:")) != -1)
    {
        switch (opt)
        {
        case 'h':
            PrintUsage(basename(argv[0]), std::cout);
            return {};

        case 'c':
            if (sscanf(optarg, "%d,%d", &cbSize.width, &cbSize.height) != 2)
            {
                throw std::invalid_argument("Error parsing checkerboard information");
            }

            // OpenCV expects number of interior vertices in the checkerboard,
            // not number of squares. Let's adjust for that.
            params.vtxCount.width  = cbSize.width - 1;
            params.vtxCount.height = cbSize.height - 1;
            break;

        case 's':
            if (sscanf(optarg, "%d", &params.searchWinSize) != 1)
            {
                throw std::invalid_argument("Error parseing search window size");
            }
            if (params.searchWinSize < 0)
            {
                throw std::invalid_argument("Search window size must be >= 0");
            }
            break;
        case '?':
            throw std::invalid_argument(std::string("Option -") + (char)optopt + " not recognized");
        }
    }

    for (int i = optind; i < argc; ++i)
    {
        params.images.push_back(argv[i]);
    }

    if (params.images.empty())
    {
        throw std::invalid_argument("At least one image must be defined");
    }

    if (cbSize.width <= 3 || cbSize.height <= 3)
    {
        throw std::invalid_argument("Checkerboard size must have at least 3x3 squares");
    }

    if (params.searchWinSize == 1)
    {
        throw std::invalid_argument("Search window size must be 0 (default) or >= 2");
    }

    return params;
}

int main(int argc, char *argv[])
{
    int retval = 0;

    VPIStream stream = NULL;
    VPIPayload remap = NULL;
    VPIImage tmpIn = NULL, tmpOut = NULL;
    VPIImage vimg = nullptr;

    try
    {
        // First parse command line paramers
        Params params = ParseParameters(argc, argv);
        if (params.images.empty()) // user just wanted the help message?
        {
            return 0;
        }

        // Where to store checkerboard 2D corners of each input image.
        std::vector<std::vector<cv::Point2f>> corners2D;

        // Store image size. All input images must have same size.
        cv::Size imgSize = {};

        for (unsigned i = 0; i < params.images.size(); ++i)
        {
            // Load input image and do some sanity check
            cv::Mat img = cv::imread(params.images[i]);
            if (img.empty())
            {
                throw std::runtime_error("Can't read " + std::string(params.images[i]));
            }

            if (imgSize == cv::Size{})
            {
                imgSize = img.size();
            }
            else if (imgSize != img.size())
            {
                throw std::runtime_error("All images must have same size");
            }

            // Find the checkerboard pattern on the image, saving the 2D
            // coordinates of checkerboard vertices in cbVertices.
            // Vertex is the point where 4 squares (2 white and 2 black) meet.
            std::vector<cv::Point2f> cbVertices;

            if (findChessboardCorners(img, params.vtxCount, cbVertices,
                                      cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE))
            {
                // Needs to perform further corner refinement?
                if (params.searchWinSize >= 2)
                {
                    cv::Mat gray;
                    cvtColor(img, gray, cv::COLOR_BGR2GRAY);

                    cornerSubPix(gray, cbVertices, cv::Size(params.searchWinSize / 2, params.searchWinSize / 2),
                                 cv::Size(-1, -1),
                                 cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.0001));
                }

                // save this image's 2D vertices in vector
                corners2D.push_back(std::move(cbVertices));
            }
            else
            {
                std::cerr << "Warning: checkerboard pattern not found in image " << params.images[i] << std::endl;
            }
        }

        // Create the vector that stores 3D coordinates for each checkerboard pattern on a space
        // where X and Y are orthogonal and run along the checkerboard sides, and Z==0 in all points on
        // checkerboard.
        std::vector<cv::Point3f> initialCheckerboard3DVertices;
        for (int i = 0; i < params.vtxCount.height; ++i)
        {
            for (int j = 0; j < params.vtxCount.width; ++j)
            {
                // since we're not interested in extrinsic camera parameters,
                // we can assume that checkerboard square size is 1x1.
                initialCheckerboard3DVertices.emplace_back(j, i, 0);
            }
        }

        // Initialize a vector with initial checkerboard positions for all images
        std::vector<std::vector<cv::Point3f>> corners3D(corners2D.size(), initialCheckerboard3DVertices);

        // Camera intrinsic parameters, initially identity (will be estimated by calibration process).
        using Mat3     = cv::Matx<double, 3, 3>;
        Mat3 camMatrix = Mat3::eye();

        // stores the fisheye model coefficients.
        std::vector<double> coeffs(4);

        // VPI currently doesn't support skew parameter on camera matrix, make sure
        // calibration process fixes it to 0.
        int flags = cv::fisheye::CALIB_FIX_SKEW;

        // Run calibration
        {
            cv::Mat rvecs, tvecs; // stores rotation and translation for each camera, not needed now.
            double rms = cv::fisheye::calibrate(corners3D, corners2D, imgSize, camMatrix, coeffs, rvecs, tvecs, flags);
            printf("rms error: %lf\n", rms);
        }

        // Output calibration result.
        printf("Fisheye coefficients: %lf %lf %lf %lf\n", coeffs[0], coeffs[1], coeffs[2], coeffs[3]);

        printf("Camera matrix:\n");
        printf("[%lf %lf %lf; %lf %lf %lf; %lf %lf %lf]\n", camMatrix(0, 0), camMatrix(0, 1), camMatrix(0, 2),
               camMatrix(1, 0), camMatrix(1, 1), camMatrix(1, 2), camMatrix(2, 0), camMatrix(2, 1), camMatrix(2, 2));

        // Now use VPI to undistort the input images:

        // Allocate a dense map.
        VPIWarpMap map            = {};
        map.grid.numHorizRegions  = 1;
        map.grid.numVertRegions   = 1;
        map.grid.regionWidth[0]   = imgSize.width;
        map.grid.regionHeight[0]  = imgSize.height;
        map.grid.horizInterval[0] = 1;
        map.grid.vertInterval[0]  = 1;
        CHECK_STATUS(vpiWarpMapAllocData(&map));

        // Initialize the fisheye lens model with the coefficients given by calibration procedure.
        VPIFisheyeLensDistortionModel distModel = {};
        distModel.mapping                       = VPI_FISHEYE_EQUIDISTANT;
        distModel.k1                            = coeffs[0];
        distModel.k2                            = coeffs[1];
        distModel.k3                            = coeffs[2];
        distModel.k4                            = coeffs[3];

        // Fill up the camera intrinsic parameters given by camera calibration procedure.
        VPICameraIntrinsic K;
        for (int i = 0; i < 2; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                K[i][j] = camMatrix(i, j);
            }
        }

        // Camera extrinsics is be identity.
        VPICameraExtrinsic X = {};
        X[0][0] = X[1][1] = X[2][2] = 1;

        // Generate a warp map to undistort an image taken from fisheye lens with
        // given parameters calculated above.
        vpiWarpMapGenerateFromFisheyeLensDistortionModel(K, X, K, &distModel, &map);

        // Create the Remap payload for undistortion given the map generated above.
        CHECK_STATUS(vpiCreateRemap(VPI_BACKEND_CUDA, &map, &remap));

        // Now that the remap payload is created, we can destroy the warp map.
        vpiWarpMapFreeData(&map);

        // Create a stream where operations will take place. We're using CUDA
        // processing.
        CHECK_STATUS(vpiStreamCreate(VPI_BACKEND_CUDA, &stream));

        // Temporary input and output images in NV12 format.
        CHECK_STATUS(vpiImageCreate(imgSize.width, imgSize.height, VPI_IMAGE_FORMAT_NV12_ER, 0, &tmpIn));
        CHECK_STATUS(vpiImageCreate(imgSize.width, imgSize.height, VPI_IMAGE_FORMAT_NV12_ER, 0, &tmpOut));

        // For each input image,
        for (unsigned i = 0; i < params.images.size(); ++i)
        {
            // Read it from disk.
            cv::Mat img = cv::imread(params.images[i]);
            assert(!img.empty());

            // Wrap it into a VPIImage
            if (vimg == nullptr)
            {
                // Now create a VPIImage that wraps it.
                CHECK_STATUS(vpiImageCreateOpenCVMatWrapper(img, 0, &vimg));
            }
            else
            {
                CHECK_STATUS(vpiImageSetWrappedOpenCVMat(vimg, img));
            }

            // Convert BGR -> NV12
            CHECK_STATUS(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_CUDA, vimg, tmpIn, NULL));

            // Undistorts the input image.
            CHECK_STATUS(vpiSubmitRemap(stream, VPI_BACKEND_CUDA, remap, tmpIn, tmpOut, VPI_INTERP_CATMULL_ROM,
                                        VPI_BORDER_ZERO, 0));

            // Convert the result NV12 back to BGR, writing back to the input image.
            CHECK_STATUS(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_CUDA, tmpOut, vimg, NULL));

            // Wait until conversion finishes.
            CHECK_STATUS(vpiStreamSync(stream));

            // Since vimg is wrapping the OpenCV image, the result is already there.
            // We just have to save it to disk.
            char buf[64];
            snprintf(buf, sizeof(buf), "undistort_%03d.jpg", i);
            imwrite(buf, img);
        }
    }
    catch (std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        PrintUsage(basename(argv[0]), std::cerr);

        retval = 1;
    }

    vpiStreamDestroy(stream);
    vpiPayloadDestroy(remap);
    vpiImageDestroy(tmpIn);
    vpiImageDestroy(tmpOut);
    vpiImageDestroy(vimg);

    return retval;
}
