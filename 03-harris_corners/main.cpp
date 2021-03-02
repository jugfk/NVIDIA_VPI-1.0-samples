/*
* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
#    include <opencv2/contrib/contrib.hpp> // for applyColorMap
#    include <opencv2/highgui/highgui.hpp>
#endif

#include <opencv2/imgproc/imgproc.hpp>
#include <vpi/OpenCVInterop.hpp>

#include <vpi/Array.h>
#include <vpi/Image.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/HarrisCorners.h>

#include <cstdio>
#include <cstring> // for memset
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

static cv::Mat DrawKeypoints(cv::Mat img, VPIKeypoint *kpts, uint32_t *scores, int numKeypoints)
{
    cv::Mat out;
    img.convertTo(out, CV_8UC1);
    cvtColor(out, out, cv::COLOR_GRAY2BGR);

    if (numKeypoints == 0)
    {
        return out;
    }

    // prepare our colormap
    cv::Mat cmap(1, 256, CV_8UC3);
    {
        cv::Mat gray(1, 256, CV_8UC1);
        for (int i = 0; i < 256; ++i)
        {
            gray.at<unsigned char>(0, i) = i;
        }
        applyColorMap(gray, cmap, cv::COLORMAP_HOT);
    }

    float maxScore = *std::max_element(scores, scores + numKeypoints);

    for (int i = 0; i < numKeypoints; ++i)
    {
        cv::Vec3b color = cmap.at<cv::Vec3b>(scores[i] / maxScore * 255);
        circle(out, cv::Point(kpts[i].x, kpts[i].y), 3, cv::Scalar(color[0], color[1], color[2]), -1);
    }

    return out;
}

int main(int argc, char *argv[])
{
    VPIImage image     = NULL;
    VPIArray keypoints = NULL;
    VPIArray scores    = NULL;
    VPIStream stream   = NULL;
    VPIPayload harris  = NULL;

    int retval = 0;

    try
    {
        if (argc != 3)
        {
            throw std::runtime_error(std::string("Usage: ") + argv[0] + " <cpu|pva|cuda> <input image>");
        }

        std::string strBackend       = argv[1];
        std::string strInputFileName = argv[2];

        // Load the input image
        cv::Mat cvImage = cv::imread(strInputFileName, cv::IMREAD_GRAYSCALE);
        if (cvImage.empty())
        {
            throw std::runtime_error("Can't open '" + strInputFileName + "'");
        }

        // Currently we only accept signed 16bpp
        cvImage.convertTo(cvImage, CV_16SC1);

        // Now parse the backend
        VPIBackend backend;

        if (strBackend == "cpu")
        {
            backend = VPI_BACKEND_CPU;
        }
        else if (strBackend == "cuda")
        {
            backend = VPI_BACKEND_CUDA;
        }
        else if (strBackend == "pva")
        {
            backend = VPI_BACKEND_PVA;
        }
        else
        {
            throw std::runtime_error("Backend '" + strBackend +
                                     "' not recognized, it must be either cpu, cuda or pva.");
        }

        // Create the stream for the given backend.
        CHECK_STATUS(vpiStreamCreate(backend, &stream));

        // We now wrap the loaded image into a VPIImage object to be used by VPI.
        // VPI won't make a copy of it, so the original
        // image must be in scope at all times.
        CHECK_STATUS(vpiImageCreateOpenCVMatWrapper(cvImage, 0, &image));

        // Create the output keypoint array. Currently for PVA backend it must have 8192 elements.
        CHECK_STATUS(vpiArrayCreate(8192, VPI_ARRAY_TYPE_KEYPOINT, 0, &keypoints));

        // Create the output scores array. It also must have 8192 elements and elements must be uint32_t.
        CHECK_STATUS(vpiArrayCreate(8192, VPI_ARRAY_TYPE_U32, 0, &scores));

        // Create the payload for Harris Corners Detector algorithm
        CHECK_STATUS(vpiCreateHarrisCornerDetector(backend, cvImage.cols, cvImage.rows, &harris));

        // Submit it for processing passing the inputs and outputs.
        {
            VPIHarrisCornerDetectorParams params;
            params.gradientSize   = 5;
            params.blockSize      = 5;
            params.strengthThresh = 20;
            params.sensitivity    = 0.01;
            params.minNMSDistance = 8; // must be 8 for PVA backend

            CHECK_STATUS(vpiSubmitHarrisCornerDetector(stream, backend, harris, image, keypoints, scores, &params));
        }

        // Wait until the algorithm finishes processing
        CHECK_STATUS(vpiStreamSync(stream));

        // Now let's retrieve the output
        {
            // Lock output keypoints and scores to retrieve its data on cpu memory
            VPIArrayData outKeypointsData;
            VPIArrayData outScoresData;
            CHECK_STATUS(vpiArrayLock(keypoints, VPI_LOCK_READ, &outKeypointsData));
            CHECK_STATUS(vpiArrayLock(scores, VPI_LOCK_READ, &outScoresData));

            VPIKeypoint *outKeypoints = (VPIKeypoint *)outKeypointsData.data;
            uint32_t *outScores       = (uint32_t *)outScoresData.data;

            printf("%d keypoints found\n", *outKeypointsData.sizePointer);

            cv::Mat outImage = DrawKeypoints(cvImage, outKeypoints, outScores, *outKeypointsData.sizePointer);

            imwrite("harris_corners_" + strBackend + ".png", outImage);

            // Done handling outputs, don't forget to unlock them.
            CHECK_STATUS(vpiArrayUnlock(scores));
            CHECK_STATUS(vpiArrayUnlock(keypoints));
        }
    }
    catch (std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        retval = 1;
    }

    // Clean up

    // Make sure stream is synchronized before destroying the objects
    // that might still be in use.
    if (stream != NULL)
    {
        vpiStreamSync(stream);
    }

    vpiImageDestroy(image);
    vpiArrayDestroy(keypoints);
    vpiArrayDestroy(scores);
    vpiPayloadDestroy(harris);
    vpiStreamDestroy(stream);

    return retval;
}
