
#pragma once

#include <iostream>
#include <queue>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "camodocal/camera_models/EquidistantCamera.h"

#include "parameters.h"
#include "tic_toc.h"
#include "utility.h"

// #include <opencv2/line_descriptor.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

#define _SHITONG_ 1
#define _SHOWLINEFLOW_ 1
#define _ONELINE_ 0
#define _CONTROLLINEDENSE_ 1

#define _LAYER_ 4
#define _POINTNUM_ 5

using namespace std;
using namespace cv;
using namespace camodocal;

struct Line
{
    Point2f StartPt;
    Point2f EndPt;
    Point2f MidPt;
    Point2f MidPt1;
    Point2f MidPt2;
    // Point2f MidPt3;
    // Point2f MidPt4;
    Point2f MidPt5;
    Point2f MidPt6;

    Point2f norm_start;
    Point2f norm_end;

    // float dealtx;
    // float dealty;

    float length;

    vector<Point2f> keyPoint;

    unsigned short idInLastFrame;
    int colorIdx;

    Line()
    {
        idInLastFrame = -1;
    }

    Line(cv::Vec4f line)
    {

        StartPt = Point2f(line[0], line[1]);
        EndPt = Point2f(line[2], line[3]);
        MidPt = 0.5 * (StartPt + EndPt);
        length = sqrt((StartPt.x - EndPt.x) * (StartPt.x - EndPt.x) + (StartPt.y - EndPt.y) * (StartPt.y - EndPt.y));
        MidPt1 = 0.5 * Point2f(StartPt.x + MidPt.x, StartPt.y + MidPt.y);
        MidPt2 = 0.5 * Point2f(EndPt.x + MidPt.x, EndPt.y + MidPt.y);
        MidPt5 = 0.7 * StartPt + 0.3 * MidPt1;
        MidPt6 = 0.7 * EndPt + 0.3 * MidPt2;
        // MidPt5 = StartPt - 10 * (StartPt - MidPt) / length / 2;
        // MidPt6 = EndPt - 10 * (EndPt - MidPt) / length / 2;

        // dealtx = EndPt.x - StartPt.x;
        // dealty = EndPt.y - StartPt.y;

        keyPoint.push_back(MidPt);
        keyPoint.push_back(MidPt5);
        // keyPoint.push_back(StartPt);
        keyPoint.push_back(MidPt1);
        keyPoint.push_back(MidPt2);
        // keyPoint.push_back(MidPt3);
        // keyPoint.push_back(MidPt4);
        keyPoint.push_back(MidPt6);
        // keyPoint.push_back(EndPt);
        idInLastFrame = -1;
    }

    void resetLine()
    {
        MidPt = 0.5 * (StartPt + EndPt);
        length = sqrt((StartPt.x - EndPt.x) * (StartPt.x - EndPt.x) + (StartPt.y - EndPt.y) * (StartPt.y - EndPt.y));
        MidPt1 = 0.5 * Point2f(StartPt.x + MidPt.x, StartPt.y + MidPt.y);
        MidPt2 = 0.5 * Point2f(EndPt.x + MidPt.x, EndPt.y + MidPt.y);
        Point2f MidPt3 = 0.5 * Point2f(MidPt.x + MidPt1.x, MidPt.y + MidPt1.y);
        Point2f MidPt4 = 0.5 * Point2f(MidPt.x + MidPt2.x, MidPt.y + MidPt2.y);
        MidPt5 = 0.8 * StartPt + 0.2 * MidPt1;
        MidPt6 = 0.8 * EndPt + 0.2 * MidPt2;
        keyPoint[0] = MidPt;
        keyPoint[1] = MidPt5;
        keyPoint[2] = MidPt1;
        keyPoint[3] = MidPt2;
        keyPoint[4] = MidPt6;
    }

    bool checkBoundry(Point2f &now, const Mat &img)
    {
        if (now.x > img.cols - 1 || now.x < 1)
            return false;
        if (now.y > img.rows - 1 || now.y < 1)
            return false;
        return true;
    }

    bool extendLine(const Point2f &start, const Point2f &end, const Point2d &mid, const Mat &magnitude, const Mat &angle, int step)
    {
        int halflength = 1;
        bool ret = false;

        // ROS_WARN("begin extend\n");
        //算出扩展线短点的方向
        float lengthDir = sqrt((start.x - mid.x) * (start.x - mid.x) + (start.y - mid.y) * (start.y - mid.y));
        float startDirX = step * (start.x - mid.x) / lengthDir;
        float startDirY = step * (start.y - mid.y) / lengthDir;
        lengthDir = sqrt((end.x - mid.x) * (end.x - mid.x) + (end.y - mid.y) * (end.y - mid.y));
        float endDirX = step * (end.x - mid.x) / lengthDir;
        float endDirY = step * (end.y - mid.y) / lengthDir;

        //原线取样求梯度
        // ROS_WARN("compute sample gradient\n");
        int sampleSegLength = 4;
        int samplePointNum = length / sampleSegLength;
        float sampleDirX = sampleSegLength * (end.x - start.x) / length;
        float sampleDirY = sampleSegLength * (end.y - start.y) / length;

        float angleThreshold = atan2(sampleDirY, sampleDirX) * 180 / M_PI;
        if (angleThreshold < 0)
            angleThreshold += 180;
        // ROS_WARN("angle threshold %f", angleThreshold);

        vector<float> sampleMagnitude, sampleAngle;

        for (int i = 0; i < samplePointNum; i++)
        {
            for (int j = -halflength; j <= halflength; j++)
                for (int k = -halflength; k <= -halflength; k++)
                {
                    Point2f now = Point2f(start.x + sampleDirX * i + j, start.y + sampleDirY * i + k);
                    if (!checkBoundry(now, magnitude))
                        continue;
                    float nowAngle = angle.at<float>(now);
                    if (nowAngle > 180)
                        nowAngle -= 180;
                    sampleMagnitude.push_back(magnitude.at<float>(now));
                    sampleAngle.push_back(nowAngle);
                    // ROS_WARN("(%f %f) sampleMagnitude:%f  sampleAngle:%f\n", now.x, now.y, magnitude.at<float>(now), nowAngle);
                }
        }
        // ROS_WARN("sampleMagnitude:%d  sampleAngle:%d\n", sampleMagnitude.size(), sampleAngle.size());

        //求梯度方差、均值、最小值，梯度方向平均值
        if (sampleMagnitude.size() == 0)
            return false;

        float mean, stdev, min_, angle_mean;
        float sum = accumulate(sampleMagnitude.begin(), sampleMagnitude.end(), 0.0);
        // ROS_WARN("compute mean stdev min\n");
        mean = sum / sampleMagnitude.size();
        double accum = 0;
        std::for_each(sampleMagnitude.begin(), sampleMagnitude.end(), [&](const float d)
                      { accum += (d - accum) * (d + accum); });
        stdev = sqrt(accum / sampleMagnitude.size());
        min_ = *min_element(sampleMagnitude.begin(), sampleMagnitude.end());

        float angle_sum = accumulate(sampleAngle.begin(), sampleAngle.end(), 0.0);
        angle_mean = angle_sum / sampleAngle.size();

        //扩展线
        // ROS_WARN("extend line\n");

        // float selectMagnitude = min(mean - 2 * stdev, min_);
        float selectMagnitude = mean - 2 * stdev;
        float setAngle = 3;
        int extendPointNum = 5;
        float nowMagnitude, nowAngle;
        for (int i = 0; i < extendPointNum; i++)
        {
            nowMagnitude = 0;
            nowAngle = 0;
            Point2f now = Point2f(start.x + startDirX * i, start.y + startDirY * i);
            if (!checkBoundry(now, magnitude))
                break;
            for (int j = -halflength; j <= halflength; j++)
                for (int k = -halflength; k <= -halflength; k++)
                {
                    Point2f now = Point2f(start.x + startDirX * i + j, start.y + startDirY * i + k);
                    nowMagnitude += magnitude.at<float>(now);
                    nowAngle += angle.at<float>(now);
                }
            nowMagnitude /= (2 * halflength + 1) * (2 * halflength + 1);
            nowAngle /= (2 * halflength + 1) * (2 * halflength + 1);
            if (nowMagnitude < selectMagnitude)
                break;
            if (nowAngle > 180)
                nowAngle -= 180;
            if (abs(nowAngle - angleThreshold - 90) > setAngle &&
                abs(nowAngle - angleThreshold + 90) > setAngle)
                break;
            StartPt = now;
            ret = true;
            // ROS_WARN("start extend\n");
        }

        for (int i = 0; i < extendPointNum; i++)
        {
            nowMagnitude = 0;
            nowAngle = 0;
            Point2f now = Point2f(end.x + endDirX * i, end.y + endDirY * i);
            if (!checkBoundry(now, magnitude))
                break;
            for (int j = -halflength; j <= halflength; j++)
                for (int k = -halflength; k <= -halflength; k++)
                {
                    Point2f now = Point2f(end.x + endDirX * i + j, end.y + endDirY * i + k);
                    nowMagnitude += magnitude.at<float>(now);
                    nowAngle += angle.at<float>(now);
                }
            nowMagnitude /= (2 * halflength + 1) * (2 * halflength + 1);
            nowAngle /= (2 * halflength + 1) * (2 * halflength + 1);
            if (nowMagnitude < selectMagnitude)
                break;

            if (nowAngle > 180)
                nowAngle -= 180;
            if (abs(nowAngle - angleThreshold - 90) > setAngle &&
                abs(nowAngle - angleThreshold + 90) > setAngle)
                break;
            EndPt = now;
            ret = true;
            // ROS_WARN("end extend\n");
        }
        return ret;
    }

    void recoverLine(int success, const Mat &magnitude, const Mat &angle, int step)
    {
        // imshow("magnitude", magnitude);
        // imshow("angle", angle);
        // waitKey();
        // ROS_WARN("%d points\n",keyPoint.size());
        Point2f nowStrat = keyPoint[1];
        if (nowStrat.x < 3 || nowStrat.x > magnitude.cols - 4 ||
            nowStrat.y < 3 || nowStrat.y > magnitude.rows - 4)
        {
            nowStrat = keyPoint[1] = keyPoint[2];
            keyPoint[2] = 0.8 * keyPoint[2] + 0.2 * keyPoint[0];
        }

        Point2f nowEnd = keyPoint[keyPoint.size() - 1];
        if (nowEnd.x < 3 || nowEnd.x > magnitude.cols - 4 ||
            nowEnd.y < 3 || nowEnd.y > magnitude.rows - 4)
        {
            nowEnd = keyPoint[keyPoint.size() - 1] = keyPoint[keyPoint.size() - 2];
            keyPoint[keyPoint.size() - 2] = 0.8 * keyPoint[keyPoint.size() - 2] + 0.2 * keyPoint[0];
        }

        Point2f nowMid = 0.5 * (nowStrat + nowEnd);

        StartPt = nowStrat;
        EndPt = nowEnd;
        length = sqrt((nowStrat.x - nowEnd.x) * (nowStrat.x - nowEnd.x) + (nowStrat.y - nowEnd.y) * (nowStrat.y - nowEnd.y));
        bool ret = false;
        ret = extendLine(nowStrat, nowEnd, nowMid, magnitude, angle, step);

        //重新分布点
        if (ret)
        {
            // ROS_WARN("re distrubute point\n");
            MidPt = 0.5 * (StartPt + EndPt);
            MidPt1 = 0.5 * Point2f(StartPt.x + MidPt.x, StartPt.y + MidPt.y);
            MidPt2 = 0.5 * Point2f(EndPt.x + MidPt.x, EndPt.y + MidPt.y);

            length = sqrt((StartPt.x - EndPt.x) * (StartPt.x - EndPt.x) + (StartPt.y - EndPt.y) * (StartPt.y - EndPt.y));
            MidPt5 = StartPt - 5 * (StartPt - MidPt) / length / 2;
            MidPt6 = EndPt - 5 * (EndPt - MidPt) / length / 2;

            keyPoint[0] = MidPt;
            keyPoint[1] = MidPt5;
            keyPoint[2] = MidPt1;
            keyPoint[3] = MidPt2;
            keyPoint[4] = MidPt6;
        }
        else
        {
            StartPt = nowStrat;
            EndPt = nowEnd;
            length = sqrt((StartPt.x - EndPt.x) * (StartPt.x - EndPt.x) + (StartPt.y - EndPt.y) * (StartPt.y - EndPt.y));
        }

        idInLastFrame = success;
    }
    bool compare(Line l)
    {
        return true;
    }
};

struct LineST
{
    Point2f start;
    Point2f end;
};

typedef vector<LineST> LineRecord;

class FrameLines
{
public:
    int frame_id;
    Mat img;
    vector<Mat> img_pyr;

    vector<Line> vecLine;
    vector<LineRecord> lineRec;
    vector<int> lineID;

    vector<int> success;

    Mat magnitude;
    Mat angle;
};
typedef shared_ptr<FrameLines> FrameLinesPtr;

class LineFeatureTracker
{
public:
    LineFeatureTracker();

    void readIntrinsicParameter(const string &calib_file);
    vector<Line> &normalizedLineEndPoints();
    vector<Line> &unNormalizedLineEndPoints() { return cur_img->vecLine; }

    void readImage(const cv::Mat &_img);

    void setMask();
    void setMerge(cv::Mat &mergel);

    FrameLinesPtr cur_img, forw_img;

    cv::Mat undist_map1_, undist_map2_, K_;

    camodocal::CameraPtr m_camera; // pinhole camera

    int frame_cnt;
    vector<int> ids;           // 每个特征点的id
    vector<int> linetrack_cnt; // 记录某个特征已经跟踪多少帧了，即被多少帧看到了
    int allfeature_cnt;        // 用来统计整个地图中有了多少条线，它将用来赋值
    cv::Mat mask;
    cv::Mat merge;

    double sum_time;
    double mean_time;
};

#if _SHITONG_

//光流部分
class OpticalFlowTracker
{
public:
    OpticalFlowTracker(
        const Mat &magnitude_,
        const Mat &img1_,
        const Mat &img2_,
        const vector<Line> &kp1_,
        vector<Line> &kp2_,
        vector<int> &success_,
        bool inverse_ = true, bool has_initial_ = false, int layer_ = 1) : magnitude(magnitude_), img1(img1_), img2(img2_), kp1(kp1_), kp2(kp2_), success(success_), inverse(inverse_),
                                                                           has_initial(has_initial_), layer(layer_) {}

    // vector<vector<Point2f>> kp_2;

    void calculateOpticalFlow(const Range &range);
    void SVDcalculateOpticalFlow(const Range &range);

private:
    const Mat &img1;
    const Mat &img2;
    const vector<Line> &kp1;
    const Mat &magnitude;
    vector<Line> &kp2;
    vector<int> &success;
    bool inverse = true;
    bool has_initial = false;
    int layer;
};

void OpticalFlowSingleLevel(
    const Mat &magnitude,
    const Mat &img1,
    const Mat &img2,
    const vector<Line> &kp1,
    vector<Line> &kp2,
    vector<int> &success,
    bool inverse = false,
    bool has_initial_guess = false, int layer_ = 0);

#define _LAYERS_ 5;

void OpticalFlowMultiLevel(
    const Mat &magnitude,
    const Mat &angle,
    vector<Mat> &img1_pyr,
    vector<Mat> &img2_pyr,
    const vector<Line> &kp1,
    vector<Line> &kp2,
    vector<int> &success,
    bool inverse = false);
#endif

void drawLine(const Line &line, Mat &merge, int index);
bool mergeLine(const Line &line, const Mat &merge, vector<Line> &vecline, Mat &maskl);
bool checkBoundry(const Point2f &point, int row, int col);
