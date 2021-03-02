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

#include <vpi/Image.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/BoxFilter.h>

#include <cstring> // for memset
#include <fstream>
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
    VPIImage image   = NULL;
    VPIImage blurred = NULL;
    VPIStream stream = NULL;

    int retval = 0;

    try
    {
        if (argc != 2)
        {
            throw std::runtime_error(std::string("Usage: ") + argv[0] + " <cpu|pva|cuda>");
        }

        std::string strBackend = argv[1];

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

        char imgContents[512][512];
        for (int i = 0; i < 512; ++i)
        {
            for (int j = 0; j < 512; ++j)
            {
                imgContents[i][j] = i * 512 + j + i;
            }
        }

        // We now wrap the loaded image into a VPIImage object to be used by VPI.
        {
            // First fill VPIImageData with the, well, image data...
            VPIImageData imgData;
            memset(&imgData, 0, sizeof(imgData));
            imgData.format               = VPI_IMAGE_FORMAT_U8;
            imgData.numPlanes            = 1;
            imgData.planes[0].width      = 512;
            imgData.planes[0].height     = 512;
            imgData.planes[0].pitchBytes = 512;
            imgData.planes[0].data       = imgContents[0];

            // Wrap it into a VPIImage. VPI won't make a copy of it, so the original
            // image must be in scope at all times.
            CHECK_STATUS(vpiImageCreateHostMemWrapper(&imgData, 0, &image));
        }

        // Now create the output image, single unsigned 8-bit channel.
        CHECK_STATUS(vpiImageCreate(512, 512, VPI_IMAGE_FORMAT_U8, 0, &blurred));

        // Submit it for processing passing the image to be blurred and the result image
        CHECK_STATUS(vpiSubmitBoxFilter(stream, backend, image, blurred, 3, 3, VPI_BORDER_ZERO));

        // Wait until the algorithm finishes processing
        CHECK_STATUS(vpiStreamSync(stream));

        // Now let's retrieve the output image contents and output it to disk
        {
            // Lock output image to retrieve its data on cpu memory
            VPIImageData outData;
            CHECK_STATUS(vpiImageLock(blurred, VPI_LOCK_READ, &outData));

            std::ofstream fd(("boxfiltered_" + strBackend + ".pgm").c_str());

            fd << "P5\n512 512 255\n";
            for (int i = 0; i < 512; ++i)
            {
                fd.write(reinterpret_cast<const char *>(outData.planes[0].data) + outData.planes[0].pitchBytes * i,
                         512);
            }
            fd.close();

            // Done handling output image, don't forget to unlock it.
            CHECK_STATUS(vpiImageUnlock(blurred));
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
    vpiImageDestroy(blurred);
    vpiStreamDestroy(stream);

    return retval;
}
