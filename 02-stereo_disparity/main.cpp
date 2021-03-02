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
#    include <opencv2/highgui/highgui.hpp>
#endif

#include <opencv2/imgproc/imgproc.hpp>
#include <vpi/OpenCVInterop.hpp>

#include <vpi/Image.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/StereoDisparity.h>

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

int main(int argc, char *argv[])
{
    VPIImage left      = NULL;
    VPIImage right     = NULL;
    VPIImage disparity = NULL;
    VPIStream stream   = NULL;
    VPIPayload stereo  = NULL;

    int retval = 0;

    try
    {
        if (argc != 4)
        {
            throw std::runtime_error(std::string("Usage: ") + argv[0] + " <cpu|pva|cuda> <left image> <right image>");
        }

        std::string strBackend       = argv[1];
        std::string strLeftFileName  = argv[2];
        std::string strRightFileName = argv[3];

        // Load the input images
        cv::Mat cvImageLeft = cv::imread(strLeftFileName, cv::IMREAD_GRAYSCALE);
        if (cvImageLeft.empty())
        {
            throw std::runtime_error("Can't open '" + strLeftFileName + "'");
        }

        cv::Mat cvImageRight = cv::imread(strRightFileName, cv::IMREAD_GRAYSCALE);
        if (cvImageRight.empty())
        {
            throw std::runtime_error("Can't open '" + strRightFileName + "'");
        }

        // Currently we only accept unsigned 16bpp inputs.
        cvImageLeft.convertTo(cvImageLeft, CV_16UC1);
        cvImageRight.convertTo(cvImageRight, CV_16UC1);

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

        // We now wrap the loaded images into a VPIImage object to be used by VPI.
        // VPI won't make a copy of it, so the original image must be in scope at all times.
        CHECK_STATUS(vpiImageCreateOpenCVMatWrapper(cvImageLeft, 0, &left));
        CHECK_STATUS(vpiImageCreateOpenCVMatWrapper(cvImageRight, 0, &right));

        // Create the image where the disparity map will be stored.
        CHECK_STATUS(vpiImageCreate(cvImageLeft.cols, cvImageLeft.rows, VPI_IMAGE_FORMAT_U16, 0, &disparity));

        // Create the payload for Harris Corners Detector algorithm

        CHECK_STATUS(vpiCreateStereoDisparityEstimator(backend, cvImageLeft.cols, cvImageLeft.rows,
                                                       VPI_IMAGE_FORMAT_U16, NULL, &stereo));

        VPIStereoDisparityEstimatorParams params;
        params.windowSize   = 5;
        params.maxDisparity = 64;

        // Submit it with the input and output images
        CHECK_STATUS(vpiSubmitStereoDisparityEstimator(stream, backend, stereo, left, right, disparity, NULL, &params));

        // Wait until the algorithm finishes processing
        CHECK_STATUS(vpiStreamSync(stream));

        // Now let's retrieve the output
        {
            // Lock output to retrieve its data on cpu memory
            VPIImageData data;
            CHECK_STATUS(vpiImageLock(disparity, VPI_LOCK_READ, &data));

            // Make an OpenCV matrix out of this image
            cv::Mat cvOut(data.planes[0].height, data.planes[0].width, CV_16UC1, data.planes[0].data,
                          data.planes[0].pitchBytes);

            // Scale result and write it to disk
            double min, max;
            minMaxLoc(cvOut, &min, &max);
            cvOut.convertTo(cvOut, CV_8UC1, 255.0 / (max - min), -min);

            imwrite("disparity_" + strBackend + ".png", cvOut);

            // Done handling output, don't forget to unlock it.
            CHECK_STATUS(vpiImageUnlock(disparity));
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

    vpiImageDestroy(left);
    vpiImageDestroy(right);
    vpiImageDestroy(disparity);
    vpiPayloadDestroy(stereo);
    vpiStreamDestroy(stream);

    return retval;
}
