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

#include <vpi/OpenCVInterop.hpp>

#include <vpi/Image.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/FFT.h>

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

// Auxiliary functions to process spectrum before saving it to disk.
cv::Mat LogMagnitude(cv::Mat cpx);
cv::Mat CompleteFullHermitian(cv::Mat in, cv::Size fullSize);
cv::Mat InplaceFFTShift(cv::Mat mag);

int main(int argc, char *argv[])
{
    VPIImage image    = NULL;
    VPIImage imageF32 = NULL;
    VPIImage spectrum = NULL;
    VPIStream stream  = NULL;
    VPIPayload fft    = NULL;

    int retval = 0;

    try
    {
        if (argc != 3)
        {
            throw std::runtime_error(std::string("Usage: ") + argv[0] + " <cpu|cuda> <input image>");
        }

        std::string strBackend       = argv[1];
        std::string strInputFileName = argv[2];

        // Load the input image
        cv::Mat cvImage = cv::imread(strInputFileName, cv::IMREAD_GRAYSCALE);
        if (cvImage.empty())
        {
            throw std::runtime_error("Can't open '" + strInputFileName + "'");
        }

        assert(cvImage.type() == CV_8UC1);

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
        else
        {
            throw std::runtime_error("Backend '" + strBackend + "' not recognized, it must be either cpu or cuda.");
        }

        // Create the stream for the given backend.
        CHECK_STATUS(vpiStreamCreate(backend, &stream));

        // We now wrap the loaded image into a VPIImage object to be used by VPI.
        // VPI won't make a copy of it, so the original
        // image must be in scope at all times.
        CHECK_STATUS(vpiImageCreateOpenCVMatWrapper(cvImage, 0, &image));

        // Temporary image that holds the float version of input
        CHECK_STATUS(vpiImageCreate(cvImage.cols, cvImage.rows, VPI_IMAGE_FORMAT_F32, 0, &imageF32));

        // Now create the output image. Note that for real inputs, the output spectrum is a Hermitian
        // matrix (conjugate-symmetric), so only the non-redundant components are output, basically the
        // left half. We adjust the output width accordingly.
        CHECK_STATUS(vpiImageCreate(cvImage.cols / 2 + 1, cvImage.rows, VPI_IMAGE_FORMAT_2F32, 0, &spectrum));

        // Create the FFT payload that does real (space) to complex (frequency) transformation
        CHECK_STATUS(
            vpiCreateFFT(backend, cvImage.cols, cvImage.rows, VPI_IMAGE_FORMAT_F32, VPI_IMAGE_FORMAT_2F32, &fft));

        // Now our processing pipeline:

        // Convert image to float
        CHECK_STATUS(vpiSubmitConvertImageFormat(stream, backend, image, imageF32, NULL));

        // Submit it for processing passing the image to be gradient and the result image
        CHECK_STATUS(vpiSubmitFFT(stream, backend, fft, imageF32, spectrum, 0));

        // Wait until the algorithm finishes processing
        CHECK_STATUS(vpiStreamSync(stream));

        // Now let's retrieve the result spectrum
        {
            // Lock output image to retrieve its data on cpu memory
            VPIImageData outData;
            CHECK_STATUS(vpiImageLock(spectrum, VPI_LOCK_READ, &outData));

            assert(outData.format == VPI_IMAGE_FORMAT_2F32);

            // Wrap spectrum to be used by OpenCV
            cv::Mat cvSpectrum(outData.planes[0].height, outData.planes[0].width, CV_32FC2, outData.planes[0].data,
                               outData.planes[0].pitchBytes);

            // Process it
            cv::Mat mag = InplaceFFTShift(LogMagnitude(CompleteFullHermitian(cvSpectrum, cvImage.size())));

            // Normalize the result to fit in 8-bits
            normalize(mag, mag, 0, 255, cv::NORM_MINMAX);

            // Write to disk
            imwrite("spectrum_" + strBackend + ".png", mag);

            // Done handling output image, don't forget to unlock it.
            CHECK_STATUS(vpiImageUnlock(spectrum));
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
    vpiImageDestroy(imageF32);
    vpiImageDestroy(spectrum);
    vpiStreamDestroy(stream);

    // Payload is owned by the stream, so it's already destroyed
    // since the stream is now destroyed.

    return retval;
}

// Auxiliary functions --------------------------------

cv::Mat LogMagnitude(cv::Mat cpx)
{
    // Split spectrum into real and imaginary parts
    cv::Mat reim[2];
    assert(cpx.channels() == 2);
    split(cpx, reim);

    // Calculate the magnitude
    cv::Mat mag;
    magnitude(reim[0], reim[1], mag);

    // Convert to logarithm scale
    mag += cv::Scalar::all(1);
    log(mag, mag);
    mag = mag(cv::Rect(0, 0, mag.cols & -2, mag.rows & -2));

    return mag;
}

cv::Mat CompleteFullHermitian(cv::Mat in, cv::Size fullSize)
{
    assert(in.type() == CV_32FC2);

    cv::Mat out(fullSize, CV_32FC2);
    for (int i = 0; i < out.rows; ++i)
    {
        for (int j = 0; j < out.cols; ++j)
        {
            cv::Vec2f p;
            if (j < in.cols)
            {
                p = in.at<cv::Vec2f>(i, j);
            }
            else
            {
                p    = in.at<cv::Vec2f>((out.rows - i) % out.rows, (out.cols - j) % out.cols);
                p[1] = -p[1];
            }
            out.at<cv::Vec2f>(i, j) = p;
        }
    }

    return out;
}

cv::Mat InplaceFFTShift(cv::Mat mag)
{
    // Rearrange the quadrants of the fourier spectrum
    // so that the origin is at the image center.

    // Create a ROI for each 4 quadrants.
    int cx = mag.cols / 2;
    int cy = mag.rows / 2;
    cv::Mat qTL(mag, cv::Rect(0, 0, cx, cy));   // top-left
    cv::Mat qTR(mag, cv::Rect(cx, 0, cx, cy));  // top-right
    cv::Mat qBL(mag, cv::Rect(0, cy, cx, cy));  // bottom-left
    cv::Mat qBR(mag, cv::Rect(cx, cy, cx, cy)); // bottom-right

    // swap top-left with bottom-right quadrants
    cv::Mat tmp;
    qTL.copyTo(tmp);
    qBR.copyTo(qTL);
    tmp.copyTo(qBR);

    // swap top-right with bottom-left quadrants
    qTR.copyTo(tmp);
    qBL.copyTo(qTR);
    tmp.copyTo(qBL);

    return mag;
}
