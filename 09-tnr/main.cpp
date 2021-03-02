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
#    include <opencv2/videoio.hpp>
#else
#    include <opencv2/highgui/highgui.hpp>
#endif

#include <opencv2/imgproc/imgproc.hpp>
#include <vpi/OpenCVInterop.hpp>

#include <vpi/Image.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/TemporalNoiseReduction.h>

#include <algorithm>
#include <cstring> // for memset
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>

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
    // main return value
    int retval = 0;

    // Declare all VPI objects we'll need here so that we
    // can destroy them at the end.
    VPIStream stream     = NULL;
    VPIImage imgPrevious = NULL, imgCurrent = NULL, imgOutput = NULL;
    VPIImage frameBGR = NULL;
    VPIPayload tnr    = NULL;

    try
    {
        if (argc != 4)
        {
            throw std::runtime_error(std::string("Usage: ") + argv[0] + " <cpu|vic|cuda> <input_video> <output>");
        }

        std::string strBackend     = argv[1];
        std::string strInputVideo  = argv[2];
        std::string strOutputVideo = argv[3];

        // Load the input video
        cv::VideoCapture invid;
        if (!invid.open(strInputVideo))
        {
            throw std::runtime_error("Can't open '" + strInputVideo + "'");
        }

        // Open the output video for writing using input's characteristics
#if CV_MAJOR_VERSION >= 3
        int w      = invid.get(cv::CAP_PROP_FRAME_WIDTH);
        int h      = invid.get(cv::CAP_PROP_FRAME_HEIGHT);
        int fourcc = invid.get(cv::CAP_PROP_FOURCC);
        double fps = invid.get(cv::CAP_PROP_FPS);
#else
        int w      = invid.get(CV_CAP_PROP_FRAME_WIDTH);
        int h      = invid.get(CV_CAP_PROP_FRAME_HEIGHT);
        int fourcc = invid.get(CV_FOURCC('M', 'J', 'P', 'G'));
        double fps = invid.get(CV_CAP_PROP_FPS);
#endif

        cv::VideoWriter outVideo(strOutputVideo, fourcc, fps, cv::Size(w, h));
        if (!outVideo.isOpened())
        {
            throw std::runtime_error("Can't create output video");
        }

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
            throw std::runtime_error("Backend '" + strBackend +
                                     "' not recognized, it must be either cpu, cuda or vic.");
        }

        // PVA backend doesn't have currently Convert Image Format algorithm. We'll use CUDA
        // backend to do that.
        CHECK_STATUS(vpiStreamCreate(VPI_BACKEND_CUDA | backend, &stream));

        CHECK_STATUS(vpiImageCreate(w, h, VPI_IMAGE_FORMAT_NV12_ER, 0, &imgPrevious));
        CHECK_STATUS(vpiImageCreate(w, h, VPI_IMAGE_FORMAT_NV12_ER, 0, &imgCurrent));
        CHECK_STATUS(vpiImageCreate(w, h, VPI_IMAGE_FORMAT_NV12_ER, 0, &imgOutput));

        // Create a Temporal Noise Reduction payload configured to process NV12
        // frames under outdoor low light
        CHECK_STATUS(vpiCreateTemporalNoiseReduction(backend, w, h, VPI_IMAGE_FORMAT_NV12_ER, VPI_TNR_DEFAULT,
                                                     VPI_TNR_PRESET_OUTDOOR_MEDIUM_LIGHT, 0.5, &tnr));

        int curFrame = 0;
        cv::Mat cvFrame;
        while (invid.read(cvFrame))
        {
            printf("Frame: %d\n", ++curFrame);

            // frameBGR isn't allocated yet?
            if (frameBGR == NULL)
            {
                // Create a VPIImage that wraps the frame
                CHECK_STATUS(vpiImageCreateOpenCVMatWrapper(cvFrame, 0, &frameBGR));
            }
            else
            {
                // reuse existing VPIImage wrapper to wrap the new frame.
                CHECK_STATUS(vpiImageSetWrappedOpenCVMat(frameBGR, cvFrame));
            }

            // First convert it to NV12
            CHECK_STATUS(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_CUDA, frameBGR, imgCurrent, NULL));

            // Apply temporal noise reduction
            // For first frame, we have to pass NULL as previous frame, this will reset internal
            // state.
            CHECK_STATUS(vpiSubmitTemporalNoiseReduction(stream, 0, tnr, curFrame == 1 ? NULL : imgPrevious, imgCurrent,
                                                         imgOutput));

            // Convert output back to BGR
            CHECK_STATUS(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_CUDA, imgOutput, frameBGR, NULL));
            CHECK_STATUS(vpiStreamSync(stream));

            // Now add it to the output video stream
            VPIImageData imgdata;
            CHECK_STATUS(vpiImageLock(frameBGR, VPI_LOCK_READ, &imgdata));

            cv::Mat outFrame;
            CHECK_STATUS(vpiImageDataExportOpenCVMat(imgdata, &outFrame));
            outVideo << outFrame;

            CHECK_STATUS(vpiImageUnlock(frameBGR));

            // this iteration's output will be next's previous. Previous, which would be discarded, will be reused
            // to store next frame.
            std::swap(imgPrevious, imgOutput);
        };
    }
    catch (std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        retval = 1;
    }

    // Destroy all VPI resources
    vpiStreamDestroy(stream);
    vpiPayloadDestroy(tnr);
    vpiImageDestroy(imgPrevious);
    vpiImageDestroy(imgCurrent);
    vpiImageDestroy(imgOutput);
    vpiImageDestroy(frameBGR);

    return retval;
}
