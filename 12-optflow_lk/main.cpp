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

#include <vpi/Array.h>
#include <vpi/Image.h>
#include <vpi/Pyramid.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/GaussianPyramid.h>
#include <vpi/algo/HarrisCorners.h>
#include <vpi/algo/OpticalFlowPyrLK.h>

#include <algorithm>
#include <cstring> // for memset
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <vector>

// Max number of corners detected by harris corner algo
constexpr int MAX_HARRIS_CORNERS = 8192;

// Max number of keypoints to be tracked
constexpr int MAX_KEYPOINTS = 100;

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

// Helper function to fetch a frame from input
static cv::Mat FetchFrame(cv::VideoCapture &invid, int32_t &idxNextFrame)
{
    cv::Mat frame;
    if (!invid.read(frame))
    {
        return cv::Mat();
    }

    // We only support grayscale inputs
    if (frame.channels() == 3)
    {
        cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
    }

    assert(frame.type() == CV_8U);

    ++idxNextFrame;
    return frame;
}

static void SaveFileToDisk(cv::Mat img, cv::Mat mask, std::string baseFileName, int32_t frameCounter)
{
    // Create the output file name
    std::string fname = baseFileName;
    int ext           = fname.rfind('.');

    char buffer[512] = {};
    snprintf(buffer, sizeof(buffer) - 1, "%s_%04d%s", fname.substr(0, ext).c_str(), frameCounter,
             fname.substr(ext).c_str());

    cvtColor(img, img, cv::COLOR_GRAY2BGR);

    cv::Mat outputImage;
    add(img, mask, outputImage);

    // Finally, write frame to disk
    if (!cv::imwrite(buffer, outputImage, {cv::IMWRITE_JPEG_QUALITY, 70}))
    {
        throw std::runtime_error("Can't write to " + std::string(buffer));
    }
}

// Sort keypoints by decreasing score, and retain only the first 'max'
static void SortKeypoints(VPIArray keypoints, VPIArray scores, int max)
{
    VPIArrayData ptsData, scoresData;
    CHECK_STATUS(vpiArrayLock(keypoints, VPI_LOCK_READ_WRITE, &ptsData));
    CHECK_STATUS(vpiArrayLock(scores, VPI_LOCK_READ_WRITE, &scoresData));

    std::vector<int> indices(*ptsData.sizePointer);
    std::iota(indices.begin(), indices.end(), 0);

    // only need to sort the first 'max' keypoints
    std::partial_sort(indices.begin(), indices.begin() + max, indices.end(), [&scoresData](int a, int b) {
        uint32_t *score = reinterpret_cast<uint32_t *>(scoresData.data);
        return score[a] >= score[b]; // decreasing score order
    });

    // keep the only 'max' indexes.
    indices.resize(std::min<size_t>(indices.size(), max));

    VPIKeypoint *kptData = reinterpret_cast<VPIKeypoint *>(ptsData.data);

    // reorder the keypoints to keep the first 'max' with highest scores.
    std::vector<VPIKeypoint> kpt;
    std::transform(indices.begin(), indices.end(), std::back_inserter(kpt),
                   [kptData](int idx) { return kptData[idx]; });
    std::copy(kpt.begin(), kpt.end(), kptData);

    // update keypoint array size.
    *ptsData.sizePointer = kpt.size();

    vpiArrayUnlock(scores);
    vpiArrayUnlock(keypoints);
}

static int UpdateMask(cv::Mat &mask, const std::vector<cv::Scalar> &trackColors, VPIArray arrPrevPts,
                      VPIArray arrCurPts, VPIArray arrStatus)
{
    // Now that optical flow is completed, there are usually two approaches to take:
    // 1. Add new feature points from current frame using a feature detector such as
    //    \ref algo_harris_corners "Harris Corner Detector"
    // 2. Keep using the points that are being tracked.
    //
    // The sample app uses the valid feature point and continue to do the tracking.

    // Lock the input and output arrays to draw the tracks to the output mask.
    VPIArrayData prevPtsData, curPtsData, statusData;
    CHECK_STATUS(vpiArrayLock(arrPrevPts, VPI_LOCK_READ_WRITE, &prevPtsData));
    if (arrCurPts)
    {
        CHECK_STATUS(vpiArrayLock(arrCurPts, VPI_LOCK_READ, &curPtsData));
    }
    CHECK_STATUS(vpiArrayLock(arrStatus, VPI_LOCK_READ, &statusData));

    const VPIKeypoint *pPrevPts = (VPIKeypoint *)prevPtsData.data;
    const VPIKeypoint *pCurPts  = (VPIKeypoint *)curPtsData.data;
    const uint8_t *pStatus      = (uint8_t *)statusData.data;

    int numTrackedKeypoints = 0;
    int totKeypoints        = *prevPtsData.sizePointer;

    for (int i = 0; i < totKeypoints; i++)
    {
        // keypoint is being tracked?
        if (pStatus[i] == 0)
        {
            // draw the tracks
            cv::Point2f prevPoint{pPrevPts[i].x, pPrevPts[i].y};
            if (arrCurPts != NULL)
            {
                cv::Point2f curPoint{pCurPts[i].x, pCurPts[i].y};
                line(mask, prevPoint, curPoint, trackColors[i], 2);
                circle(mask, curPoint, 5, trackColors[i], -1);
            }
            else
            {
                circle(mask, prevPoint, 5, trackColors[i], -1);
            }

            numTrackedKeypoints++;
        }
    }

    // We're finished working with the arrays.
    CHECK_STATUS(vpiArrayUnlock(arrPrevPts));
    CHECK_STATUS(vpiArrayUnlock(arrStatus));
    if (arrCurPts)
    {
        CHECK_STATUS(vpiArrayUnlock(arrCurPts));
    }

    return numTrackedKeypoints;
}

int main(int argc, char *argv[])
{
    int retval = 0;

    VPIStream stream        = NULL;
    VPIImage imgPrevFrame   = NULL;
    VPIImage imgCurFrame    = NULL;
    VPIPyramid pyrPrevFrame = NULL, pyrCurFrame = NULL;
    VPIArray arrPrevPts = NULL, arrCurPts = NULL, arrStatus = NULL;
    VPIPayload optflow = NULL;
    VPIArray scores    = NULL;
    VPIPayload harris  = NULL;

    try
    {
        if (argc != 5)
        {
            throw std::runtime_error(std::string("Usage: ") + argv[0] +
                                     " <cpu|cuda> <input_video> <pyramid_levels> <output>");
        }

        std::string strBackend     = argv[1];
        std::string strInputVideo  = argv[2];
        int32_t pyrLevel           = std::stoi(argv[3]);
        std::string strOutputFiles = argv[4];

        // Load the input video
        cv::VideoCapture invid;
        if (!invid.open(strInputVideo))
        {
            throw std::runtime_error("Can't open '" + strInputVideo + "'");
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
        else
        {
            throw std::runtime_error("Backend '" + strBackend + "' not recognized, it must be either cpu or cuda.");
        }

        // Create the stream where processing will happen.
        CHECK_STATUS(vpiStreamCreate(0, &stream));

        // Counter for the frames
        int idxNextFrame = 0;

        // Fetch the first frame and wrap it into a VPIImage.
        // The points to be tracked will be gathered from this frame later on.
        cv::Mat cvPrevFrame = FetchFrame(invid, idxNextFrame);
        CHECK_STATUS(vpiImageCreateOpenCVMatWrapper(cvPrevFrame, 0, &imgPrevFrame));

        // Create the current frame wrapper using the first frame. This wrapper will
        // be set to point to every new frame, in the main loop.
        CHECK_STATUS(vpiImageCreateOpenCVMatWrapper(cvPrevFrame, 0, &imgCurFrame));

        VPIImageFormat imgFormat;
        CHECK_STATUS(vpiImageGetFormat(imgPrevFrame, &imgFormat));

        // Convert image to image pyramid
        CHECK_STATUS(vpiPyramidCreate(cvPrevFrame.cols, cvPrevFrame.rows, imgFormat, pyrLevel, 0.5, 0, &pyrPrevFrame));
        CHECK_STATUS(vpiPyramidCreate(cvPrevFrame.cols, cvPrevFrame.rows, imgFormat, pyrLevel, 0.5, 0, &pyrCurFrame));

        // Create input and output arrays
        CHECK_STATUS(vpiArrayCreate(MAX_HARRIS_CORNERS, VPI_ARRAY_TYPE_KEYPOINT, 0, &arrPrevPts));
        CHECK_STATUS(vpiArrayCreate(MAX_HARRIS_CORNERS, VPI_ARRAY_TYPE_KEYPOINT, 0, &arrCurPts));
        CHECK_STATUS(vpiArrayCreate(MAX_HARRIS_CORNERS, VPI_ARRAY_TYPE_U8, 0, &arrStatus));

        // Create Optical Flow payload
        CHECK_STATUS(
            vpiCreateOpticalFlowPyrLK(backend, cvPrevFrame.cols, cvPrevFrame.rows, imgFormat, pyrLevel, 0.5, &optflow));

        // Parameters we'll use. No need to change them on the fly, so just define them here.
        VPIOpticalFlowPyrLKParams lkParams;
        memset(&lkParams, 0, sizeof(lkParams));
        lkParams.useInitialFlow  = false;
        lkParams.termination     = VPI_TERMINATION_CRITERIA_ITERATIONS | VPI_TERMINATION_CRITERIA_EPSILON;
        lkParams.epsilonType     = VPI_LK_ERROR_L1;
        lkParams.epsilon         = 0.0f;
        lkParams.windowDimension = 15;
        lkParams.numIterations   = 6;

        // Create a mask image for drawing purposes
        cv::Mat mask = cv::Mat::zeros(cvPrevFrame.size(), CV_8UC3);

        // Create some random colors
        std::vector<cv::Scalar> trackColors;
        {
            std::vector<cv::Vec3b> tmpTrackColors;
            cv::RNG rng;
            for (int i = 0; i < MAX_KEYPOINTS; i++)
            {
                int r = rng.uniform(0, 256);
                tmpTrackColors.push_back(cv::Vec3b(r, 255, 255));
            }
            cvtColor(tmpTrackColors, tmpTrackColors, cv::COLOR_HSV2BGR);

            for (int i = 0; i < MAX_KEYPOINTS; i++)
            {
                trackColors.push_back(cv::Scalar(tmpTrackColors[i]));
            }
        }

        // Gather feature points from first frame using Harris Corners.
        {
            CHECK_STATUS(vpiArrayCreate(MAX_HARRIS_CORNERS, VPI_ARRAY_TYPE_U32, 0, &scores));

            VPIHarrisCornerDetectorParams harrisParams;
            memset(&harrisParams, 0, sizeof(harrisParams));

            CHECK_STATUS(vpiCreateHarrisCornerDetector(backend, cvPrevFrame.cols, cvPrevFrame.rows, &harris));

            harrisParams.gradientSize   = 5;
            harrisParams.blockSize      = 5;
            harrisParams.sensitivity    = 0.01;
            harrisParams.minNMSDistance = 8;

            CHECK_STATUS(
                vpiSubmitHarrisCornerDetector(stream, 0, harris, imgPrevFrame, arrPrevPts, scores, &harrisParams));

            CHECK_STATUS(vpiStreamSync(stream));

            SortKeypoints(arrPrevPts, scores, MAX_KEYPOINTS);
        }

        // Update the mask with info from first frame.
        UpdateMask(mask, trackColors, arrPrevPts, NULL, arrStatus);

        // Save first frame to disk
        SaveFileToDisk(cvPrevFrame, mask, strOutputFiles, idxNextFrame - 1);

        // Now that initialization is finished, we start the actual processing stage.

        // Generate pyramid for first frame.
        CHECK_STATUS(vpiSubmitGaussianPyramidGenerator(stream, backend, imgPrevFrame, pyrPrevFrame));

        // Fetch a new frame until video ends
        cv::Mat cvCurFrame;
        while ((cvCurFrame = FetchFrame(invid, idxNextFrame)).data != NULL)
        {
            // Wrap frame into a VPIImage, reusing the existing imgCurFrame.
            CHECK_STATUS(vpiImageSetWrappedOpenCVMat(imgCurFrame, cvCurFrame));

            // Generate pyramid
            CHECK_STATUS(vpiSubmitGaussianPyramidGenerator(stream, backend, imgCurFrame, pyrCurFrame));

            // Estimate the features' position in current frame given their position in previous frame
            CHECK_STATUS(vpiSubmitOpticalFlowPyrLK(stream, 0, optflow, pyrPrevFrame, pyrCurFrame, arrPrevPts, arrCurPts,
                                                   arrStatus, &lkParams));

            // Wait for processing to finish.
            CHECK_STATUS(vpiStreamSync(stream));

            // Update the output mask
            int numTrackedKeypoints = UpdateMask(mask, trackColors, arrPrevPts, arrCurPts, arrStatus);

            // No more keypoints being tracked?
            if (numTrackedKeypoints == 0)
            {
                break; // we can finish procesing.
            }

            printf("Frame id=%d: %d points tracked. \n", idxNextFrame, numTrackedKeypoints);

            // Save frame to disk
            SaveFileToDisk(cvCurFrame, mask, strOutputFiles, idxNextFrame - 1);

            // Swap previous frame and next frame
            std::swap(cvPrevFrame, cvCurFrame);
            std::swap(imgPrevFrame, imgCurFrame);
            std::swap(pyrPrevFrame, pyrCurFrame);
            std::swap(arrPrevPts, arrCurPts);
        }
    }
    catch (std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        retval = 1;
    }

    vpiStreamDestroy(stream);
    vpiPayloadDestroy(harris);
    vpiPayloadDestroy(optflow);

    vpiPyramidDestroy(pyrPrevFrame);
    vpiImageDestroy(imgPrevFrame);
    vpiImageDestroy(imgCurFrame);
    vpiArrayDestroy(arrPrevPts);
    vpiArrayDestroy(arrCurPts);
    vpiArrayDestroy(arrStatus);
    vpiArrayDestroy(scores);

    return retval;
}
