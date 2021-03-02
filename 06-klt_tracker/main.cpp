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
#    include <opencv2/videoio.hpp>
#else
#    include <opencv2/highgui/highgui.hpp>
#endif

#include <opencv2/imgproc/imgproc.hpp>
#include <vpi/OpenCVInterop.hpp>

#include <vpi/Array.h>
#include <vpi/Image.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/KLTFeatureTracker.h>

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

// Utility to draw the bounding boxes into an image and save it to disk.
static cv::Mat WriteKLTBoxes(VPIImage img, VPIArray boxes, VPIArray preds)
{
    // Convert img into a cv::Mat
    cv::Mat out;
    {
        VPIImageData imgdata;
        CHECK_STATUS(vpiImageLock(img, VPI_LOCK_READ, &imgdata));

        int cvtype;
        switch (imgdata.format)
        {
        case VPI_IMAGE_FORMAT_U8:
            cvtype = CV_8U;
            break;

        case VPI_IMAGE_FORMAT_S8:
            cvtype = CV_8S;
            break;

        case VPI_IMAGE_FORMAT_U16:
            cvtype = CV_16UC1;
            break;

        case VPI_IMAGE_FORMAT_S16:
            cvtype = CV_16SC1;
            break;

        default:
            throw std::runtime_error("Image type not supported");
        }

        cv::Mat cvimg(imgdata.planes[0].height, imgdata.planes[0].width, cvtype, imgdata.planes[0].data,
                      imgdata.planes[0].pitchBytes);

        if (cvimg.type() == CV_16U)
        {
            cvimg.convertTo(out, CV_8U);
            cvimg = out;
            out   = cv::Mat();
        }

        cvtColor(cvimg, out, cv::COLOR_GRAY2BGR);

        CHECK_STATUS(vpiImageUnlock(img));
    }

    // Now draw the bounding boxes.
    VPIArrayData boxdata;
    CHECK_STATUS(vpiArrayLock(boxes, VPI_LOCK_READ, &boxdata));

    VPIArrayData preddata;
    CHECK_STATUS(vpiArrayLock(preds, VPI_LOCK_READ, &preddata));

    auto *pboxes = reinterpret_cast<VPIKLTTrackedBoundingBox *>(boxdata.data);
    auto *ppreds = reinterpret_cast<VPIHomographyTransform2D *>(preddata.data);

    srand(0);
    for (int i = 0; i < *boxdata.sizePointer; ++i)
    {
        if (pboxes[i].trackingStatus == 1)
        {
            // So that the colors assigned to bounding boxes don't change
            // when some bbox isn't tracked anymore.
            rand();
            rand();
            rand();
            continue;
        }

        float x, y, w, h;
        x = pboxes[i].bbox.xform.mat3[0][2] + ppreds[i].mat3[0][2];
        y = pboxes[i].bbox.xform.mat3[1][2] + ppreds[i].mat3[1][2];
        w = pboxes[i].bbox.width * pboxes[i].bbox.xform.mat3[0][0] * ppreds[i].mat3[0][0];
        h = pboxes[i].bbox.height * pboxes[i].bbox.xform.mat3[1][1] * ppreds[i].mat3[1][1];

        rectangle(out, cv::Rect(x, y, w, h), cv::Scalar(rand() % 256, rand() % 256, rand() % 256), 2);
    }

    CHECK_STATUS(vpiArrayUnlock(preds));
    CHECK_STATUS(vpiArrayUnlock(boxes));

    return out;
}

int main(int argc, char *argv[])
{
    int retval = 0;

    // Arrays that will store our input bboxes and predicted transform.
    VPIArray inputBoxList = NULL, inputPredList = NULL;

    VPIStream stream         = NULL;
    VPIArray outputBoxList   = NULL;
    VPIArray outputEstimList = NULL;
    VPIPayload klt           = NULL;
    VPIImage imgReference    = NULL;
    VPIImage imgTemplate     = NULL;

    try
    {
        if (argc != 5)
        {
            throw std::runtime_error(std::string("Usage: ") + argv[0] +
                                     " <cpu|pva|cuda> <input_video> <bbox descr> <output>");
        }

        std::string strBackend     = argv[1];
        std::string strInputVideo  = argv[2];
        std::string strInputBBoxes = argv[3];
        std::string strOutputVideo = argv[4];

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

        // Load the bounding boxes
        // Format is: <frame number> <bbox_x> <bbox_y> <bbox_width> <bbox_height>
        // Important assumption: bboxes must be sorted with increasing frame numbers.

        // These arrays will actually wrap these vectors.
        std::vector<VPIKLTTrackedBoundingBox> bboxes;
        int32_t bboxesSize = 0;
        std::vector<VPIHomographyTransform2D> preds;
        int32_t predsSize = 0;

        // Stores how many bboxes there are in each frame. Only
        // stores when the bboxes count change.
        std::map<int, size_t> bboxes_size_at_frame; // frame -> bbox count

        // PVA requires that array capacity is 128.
        bboxes.reserve(128);
        preds.reserve(128);

        // Read bounding boxes
        {
            std::ifstream in(strInputBBoxes);
            if (!in)
            {
                throw std::runtime_error("Can't open '" + strInputBBoxes + "'");
            }

            // For each bounding box,
            int frame, x, y, w, h;
            while (in >> frame >> x >> y >> w >> h)
            {
                if (bboxes.size() == 64)
                {
                    throw std::runtime_error("Too many bounding boxes");
                }

                // Convert the axis-aligned bounding box into our tracking
                // structure.

                VPIKLTTrackedBoundingBox track = {};
                // scale
                track.bbox.xform.mat3[0][0] = 1;
                track.bbox.xform.mat3[1][1] = 1;
                // position
                track.bbox.xform.mat3[0][2] = x;
                track.bbox.xform.mat3[1][2] = y;
                // must be 1
                track.bbox.xform.mat3[2][2] = 1;

                track.bbox.width     = w;
                track.bbox.height    = h;
                track.trackingStatus = 0; // valid tracking
                track.templateStatus = 1; // must update

                bboxes.push_back(track);

                // Identity predicted transform.
                VPIHomographyTransform2D xform = {};
                xform.mat3[0][0]               = 1;
                xform.mat3[1][1]               = 1;
                xform.mat3[2][2]               = 1;
                preds.push_back(xform);

                bboxes_size_at_frame[frame] = bboxes.size();
            }

            if (!in && !in.eof())
            {
                throw std::runtime_error("Can't parse bounding boxes, stopped at bbox #" +
                                         std::to_string(bboxes.size()));
            }

            // Wrap the input arrays into VPIArray's
            VPIArrayData data = {};
            data.format       = VPI_ARRAY_TYPE_KLT_TRACKED_BOUNDING_BOX;
            data.capacity     = bboxes.capacity();
            data.sizePointer  = &bboxesSize;
            data.data         = &bboxes[0];
            CHECK_STATUS(vpiArrayCreateHostMemWrapper(&data, 0, &inputBoxList));

            data.format      = VPI_ARRAY_TYPE_HOMOGRAPHY_TRANSFORM_2D;
            data.sizePointer = &predsSize;
            data.data        = &preds[0];
            CHECK_STATUS(vpiArrayCreateHostMemWrapper(&data, 0, &inputPredList));
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

        // Helper function to fetch a frame from input
        int nextFrame   = 0;
        auto fetchFrame = [&invid, &nextFrame, backend]() {
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

            if (backend == VPI_BACKEND_PVA)
            {
                // PVA only supports 16-bit unsigned inputs,
                // where each element is in 0-255 range, so
                // no rescaling needed.
                cv::Mat aux;
                frame.convertTo(aux, CV_16U);
                frame = aux;
            }
            else
            {
                assert(frame.type() == CV_8U);
            }

            ++nextFrame;
            return frame;
        };

        // Fetch the first frame and wrap it into a VPIImage.
        // Templates will be based on this frame.
        cv::Mat cvTemplate = fetchFrame();
        CHECK_STATUS(vpiImageCreateOpenCVMatWrapper(cvTemplate, 0, &imgTemplate));

        // Create the reference image wrapper. Let's wrap the cvTemplate for now just
        // to create the wrapper. Later we'll set it to wrap the actual reference image.
        CHECK_STATUS(vpiImageCreateOpenCVMatWrapper(cvTemplate, 0, &imgReference));

        VPIImageFormat imgFormat;
        CHECK_STATUS(vpiImageGetFormat(imgTemplate, &imgFormat));

        // Using this first frame's characteristics, create a KLT Bounding Box Tracker payload.
        // We're limiting the template dimensions to 64x64.
        CHECK_STATUS(vpiCreateKLTFeatureTracker(backend, cvTemplate.cols, cvTemplate.rows, imgFormat, NULL, &klt));

        // Parameters we'll use. No need to change them on the fly, so just define them here.
        VPIKLTFeatureTrackerParams params = {};
        params.numberOfIterationsScaling  = 20;
        params.nccThresholdUpdate         = 0.8f;
        params.nccThresholdKill           = 0.6f;
        params.nccThresholdStop           = 1.0f;
        params.maxScaleChange             = 0.2f;
        params.maxTranslationChange       = 1.5f;
        params.trackingType               = VPI_KLT_INVERSE_COMPOSITIONAL;

        // Output array with estimated bbox for current frame.
        CHECK_STATUS(vpiArrayCreate(128, VPI_ARRAY_TYPE_KLT_TRACKED_BOUNDING_BOX, 0, &outputBoxList));

        // Output array with estimated transform of input bbox to match output bbox.
        CHECK_STATUS(vpiArrayCreate(128, VPI_ARRAY_TYPE_HOMOGRAPHY_TRANSFORM_2D, 0, &outputEstimList));

        cv::Mat cvReference;

        size_t curNumBoxes = 0;

        do
        {
            size_t curFrame = nextFrame - 1;

            // Get the number of bounding boxes in current frame.
            auto tmp          = --bboxes_size_at_frame.upper_bound(curFrame);
            size_t bbox_count = tmp->second;

            assert(bbox_count >= curNumBoxes && "input bounding boxes must be sorted by frame");

            // Does current frame have new bounding boxes?
            if (curNumBoxes != bbox_count)
            {
                // Update the input array sizes, the new frame is already there as we populated
                // these arrays with all input bounding boxes.
                CHECK_STATUS(vpiArrayLock(inputBoxList, VPI_LOCK_READ_WRITE, NULL));
                CHECK_STATUS(vpiArraySetSize(inputBoxList, bbox_count));
                CHECK_STATUS(vpiArrayUnlock(inputBoxList));

                CHECK_STATUS(vpiArrayLock(inputPredList, VPI_LOCK_READ_WRITE, NULL));
                CHECK_STATUS(vpiArraySetSize(inputPredList, bbox_count));
                CHECK_STATUS(vpiArrayUnlock(inputPredList));

                for (size_t i = 0; i < bbox_count - curNumBoxes; ++i)
                {
                    std::cout << curFrame << " -> new " << curNumBoxes + i << std::endl;
                }
                assert(bbox_count <= bboxes.capacity());
                assert(bbox_count <= preds.capacity());

                curNumBoxes = bbox_count;
            }

            // Save this frame to disk.
            outVideo << WriteKLTBoxes(imgTemplate, inputBoxList, inputPredList);

            // Fetch a new frame
            cvReference = fetchFrame();

            // Video ended?
            if (cvReference.data == NULL)
            {
                // Just end gracefully.
                break;
            }

            // Make the reference wrapper point to the reference frame
            CHECK_STATUS(vpiImageSetWrappedOpenCVMat(imgReference, cvReference));

            // Estimate the bounding boxes in current frame (reference) given their position in previous
            // frame (template).
            CHECK_STATUS(vpiSubmitKLTFeatureTracker(stream, backend, klt, imgTemplate, inputBoxList, inputPredList,
                                                    imgReference, outputBoxList, outputEstimList, &params));

            // Wait for processing to finish.
            CHECK_STATUS(vpiStreamSync(stream));

            // Now we lock the output arrays to properly set up the input for the next iteration.
            VPIArrayData updatedBBoxData;
            CHECK_STATUS(vpiArrayLock(outputBoxList, VPI_LOCK_READ, &updatedBBoxData));

            VPIArrayData estimData;
            CHECK_STATUS(vpiArrayLock(outputEstimList, VPI_LOCK_READ, &estimData));

            auto *updated_bbox = reinterpret_cast<VPIKLTTrackedBoundingBox *>(updatedBBoxData.data);
            auto *estim        = reinterpret_cast<VPIHomographyTransform2D *>(estimData.data);

            // For each bounding box,
            for (size_t b = 0; b < curNumBoxes; ++b)
            {
                // Did tracking failed?
                if (updated_bbox[b].trackingStatus)
                {
                    // Do we have to update the input bbox's tracking status too?
                    if (bboxes[b].trackingStatus == 0)
                    {
                        std::cout << curFrame << " -> dropped " << b << std::endl;
                        bboxes[b].trackingStatus = 1;
                    }

                    continue;
                }

                // Must update template for this bounding box??
                if (updated_bbox[b].templateStatus)
                {
                    std::cout << curFrame << " -> update " << b << std::endl;

                    // There are usually two approaches here:
                    // 1. Redefine the bounding box using a feature detector such as
                    //    \ref algo_harris_corners "Harris keypoint detector", or
                    // 2. Use updated_bbox[b], which is still valid, although tracking
                    //    errors might accumulate over time.
                    //
                    // We'll go to the second option, less robust, but simple enough
                    // to implement.
                    bboxes[b] = updated_bbox[b];

                    // Signal the input that the template for this bounding box must be updated.
                    bboxes[b].templateStatus = 1;

                    // Predicted transform is now identity as we reset the tracking.
                    preds[b]            = VPIHomographyTransform2D{};
                    preds[b].mat3[0][0] = 1;
                    preds[b].mat3[1][1] = 1;
                    preds[b].mat3[2][2] = 1;
                }
                else
                {
                    // Inform that the template for this bounding box doesn't need to be pdated.
                    bboxes[b].templateStatus = 0;

                    // We just update the input transform with the estimated one.
                    preds[b] = estim[b];
                }
            }

            // We're finished working with the output arrays.
            CHECK_STATUS(vpiArrayUnlock(outputBoxList));
            CHECK_STATUS(vpiArrayUnlock(outputEstimList));

            // Since we've updated the input arrays, tell VPI to invalidate
            // any internal buffers that might still refer to the old data.
            CHECK_STATUS(vpiArrayInvalidate(inputBoxList));
            CHECK_STATUS(vpiArrayInvalidate(inputPredList));

            // Next's reference frame is current's template.
            std::swap(imgTemplate, imgReference);
            std::swap(cvTemplate, cvReference);
        } while (true);
    }
    catch (std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        retval = 1;
    }

    vpiStreamDestroy(stream);
    vpiPayloadDestroy(klt);
    vpiArrayDestroy(inputBoxList);
    vpiArrayDestroy(inputPredList);
    vpiArrayDestroy(outputBoxList);
    vpiArrayDestroy(outputEstimList);
    vpiImageDestroy(imgReference);
    vpiImageDestroy(imgTemplate);

    return retval;
}
