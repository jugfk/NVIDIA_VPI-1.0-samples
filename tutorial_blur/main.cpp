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

// For image I/O
#include <opencv2/core/version.hpp>
#if CV_MAJOR_VERSION >= 3
#    include <opencv2/imgcodecs.hpp>
#else
#    include <opencv2/highgui/highgui.hpp>
#endif

#include <iostream>

// All vpi headers are under directory vpi/
#include <vpi/OpenCVInterop.hpp>

#include <vpi/Image.h>
#include <vpi/Stream.h>

// Algorithm we'll need. Blurring will be performed by a box filter.
#include <vpi/algo/BoxFilter.h>

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cerr << "Must pass an input image to be blurred" << std::endl;
        return 1;
    }

    // Phase 1: Initialization ---------------------------------

    // First load the input image
    cv::Mat cvImage = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (cvImage.data == NULL)
    {
        std::cerr << "Can't open input image" << std::endl;
        return 2;
    }

    // Now create the stream. Passing 0 allows algorithms submitted to it to run
    // in any available backend.
    VPIStream stream;
    vpiStreamCreate(0, &stream);

    // Now wrap the loaded image into a VPIImage object to be used by VPI.
    VPIImage image;
    vpiImageCreateOpenCVMatWrapper(cvImage, 0, &image);

    // Now create the output images, single unsigned 8-bit channel. The image lifetime is
    // managed by VPI.
    VPIImage blurred;
    vpiImageCreate(cvImage.cols, cvImage.rows, VPI_IMAGE_FORMAT_U8, 0, &blurred);

    // Phase 2: main processing --------------------------------------

    // Submit the algorithm task for processing along with inputs and outputs
    // Parameters are:
    // 1. the stream on which the algorithm will run
    // 2. Which hardware backend will execute it. Here CUDA is specified, but it could be CPU or
    //    PVA (on Jetson Xavier devices). No further changes to the program are needed to have it executed
    //    on a different hardware.
    // 3. input image to be blurred.
    // 4. output image with the result.
    // 5 and 6. box filter size, in this case, 5x5
    // 7. Border extension for when the algorithm tries to sample pixels outside the image border.
    //    VPI_BORDER_ZERO will consider all such pixels as being 0.
    vpiSubmitBoxFilter(stream, VPI_BACKEND_CUDA, image, blurred, 5, 5, VPI_BORDER_ZERO);

    // Block the current thread until until the stream finishes all tasks submitted to it up till now.
    vpiStreamSync(stream);

    // Finally retrieve the output image contents and output it to disk

    // Lock output image to retrieve its data from cpu memory
    VPIImageData outData;
    vpiImageLock(blurred, VPI_LOCK_READ, &outData);

    // Construct an cv::Mat out of the retrieved data.
    cv::Mat cvOut(outData.planes[0].height, outData.planes[0].width, CV_8UC1, outData.planes[0].data,
                  outData.planes[0].pitchBytes);
    // Just write it to disk
    imwrite("tutorial_blurred.png", cvOut);

    // Done handling output image, don't forget to unlock it.
    vpiImageUnlock(blurred);

    // Stage 3: clean up --------------------------------------

    // Destroy all created objects.
    vpiStreamDestroy(stream);
    vpiImageDestroy(image);
    vpiImageDestroy(blurred);

    return 0;
}
