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

#include <vpi/OpenCVInterop.hpp>

#include <vpi/Image.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/Rescale.h>

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
    VPIImage image      = NULL;
    VPIImage imageNV12  = NULL;
    VPIImage outputNV12 = NULL;
    VPIImage output     = NULL;
    VPIStream stream    = NULL;

    int retval = 0;

    try
    {
        if (argc != 3)
        {
            throw std::runtime_error(std::string("Usage: ") + argv[0] + " <cpu|vic|cuda> <input image>");
        }

        std::string strBackend       = argv[1];
        std::string strInputFileName = argv[2];

        // Load the input image
        cv::Mat cvImage = cv::imread(strInputFileName);
        if (cvImage.empty())
        {
            throw std::runtime_error("Can't open '" + strInputFileName + "'");
        }

        assert(cvImage.type() == CV_8UC3);

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
        else if (strBackend == "vic")
        {
            backend = VPI_BACKEND_VIC;
        }
        else
        {
            throw std::runtime_error("Backend '" + strBackend + "' not recognized, it must be either cpu, cuda or vic");
        }

        // 1. Initialization phase ---------------------------------------

        // Create the stream for the given backend. We'll also enable CUDA for gaussian filter.
        CHECK_STATUS(vpiStreamCreate(backend | VPI_BACKEND_CUDA, &stream));

        // We now wrap the loaded image into a VPIImage object to be used by VPI.
        // VPI won't make a copy of it, so the original
        // image must be in scope at all times.
        CHECK_STATUS(vpiImageCreateOpenCVMatWrapper(cvImage, 0, &image));

        // Create a temporary image to hold the input converted to NV12.
        CHECK_STATUS(vpiImageCreate(cvImage.cols, cvImage.rows, VPI_IMAGE_FORMAT_NV12_ER, 0, &imageNV12));

        // Now create the output image.
        CHECK_STATUS(vpiImageCreate(cvImage.cols / 2, cvImage.rows / 3, VPI_IMAGE_FORMAT_NV12_ER, 0, &outputNV12));

        // And the output image converted back to BGR8
        CHECK_STATUS(vpiImageCreate(cvImage.cols / 2, cvImage.rows / 3, VPI_IMAGE_FORMAT_BGR8, 0, &output));

        // 2. Computation phase ---------------------------------------

        // Convert input from BGR8 to NV12
        CHECK_STATUS(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_CUDA, image, imageNV12, NULL));

        // Now we downsample
        CHECK_STATUS(vpiSubmitRescale(stream, backend, imageNV12, outputNV12, VPI_INTERP_LINEAR, VPI_BORDER_CLAMP, 0));

        // Finally, convert the result back to BGR8
        CHECK_STATUS(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_CUDA, outputNV12, output, NULL));

        // Wait until the algorithm finishes processing
        CHECK_STATUS(vpiStreamSync(stream));

        // Now let's retrieve the output image contents and output it to disk
        {
            VPIImageData data;
            CHECK_STATUS(vpiImageLock(output, VPI_LOCK_READ, &data));

            // Lock output image to retrieve its data on cpu memory
            VPIImageData outData;
            CHECK_STATUS(vpiImageLock(output, VPI_LOCK_READ, &outData));

            cv::Mat cvOut(outData.planes[0].height, outData.planes[0].width, CV_8UC3, outData.planes[0].data,
                          outData.planes[0].pitchBytes);
            imwrite("scaled_" + strBackend + ".png", cvOut);

            // Done handling output image, don't forget to unlock it.
            CHECK_STATUS(vpiImageUnlock(output));
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
    vpiImageDestroy(imageNV12);
    vpiImageDestroy(output);
    vpiStreamDestroy(stream);

    return retval;
}
