#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>

#include "linefeature_tracker.h"

// #include "feature_tracker.h"

#define SHOW_UNDISTORTION 0

vector<uchar> r_status;
vector<float> r_err;
queue<sensor_msgs::ImageConstPtr> img_buf;
// Eigen::Matrix3d rci;

ros::Publisher pub_img, pub_match, pub_detect;

LineFeatureTracker trackerData;
double first_image_time;
int pub_count = 1;
bool first_image_flag = true;
double frame_cnt = 0;
double sum_time = 0.0;
double mean_time = 0.0;
bool init_pub = 0;

void img_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    if (first_image_flag)
    {
        first_image_flag = false;
        first_image_time = img_msg->header.stamp.toSec();
    }

    // frequency control, 如果图像频率低于一个值
    if (round(1.0 * pub_count / (img_msg->header.stamp.toSec() - first_image_time)) <= FREQ)
    {
        PUB_THIS_FRAME = true;
        // reset the frequency control
        if (abs(1.0 * pub_count / (img_msg->header.stamp.toSec() - first_image_time) - FREQ) < 0.01 * FREQ)
        {
            first_image_time = img_msg->header.stamp.toSec();
            pub_count = 0;
        }
    }
    else
        PUB_THIS_FRAME = false;

    TicToc t_r;

    if (PUB_THIS_FRAME)
    {
        cv_bridge::CvImageConstPtr ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);
        cv::Mat show_img = ptr->image;
        //    cv::imshow("lineimg",show_img);
        //    cv::waitKey(1);

        frame_cnt++;
        trackerData.readImage(ptr->image.rowRange(0, ROW)); // rowRange(i,j) 取图像的i～j行

        pub_count++;

        sensor_msgs::PointCloudPtr feature_lines(new sensor_msgs::PointCloud);
        sensor_msgs::ChannelFloat32 id_of_line;    //  feature id
        sensor_msgs::ChannelFloat32 u_of_endpoint; //  u
        sensor_msgs::ChannelFloat32 v_of_endpoint; //  v

        feature_lines->header = img_msg->header;
        feature_lines->header.frame_id = "world";

        vector<set<int>> hash_ids(NUM_OF_CAM);
        for (int i = 0; i < NUM_OF_CAM; i++)
        {

            if (i != 1 || !STEREO_TRACK) // 单目
            {
                auto &un_lines = trackerData.normalizedLineEndPoints();

                // auto &cur_lines = trackerData.cur_img->vecLine;
                auto &ids = trackerData.cur_img->lineID;

                for (unsigned int j = 0; j < ids.size(); j++)
                {

                    int p_id = ids[j];
                    hash_ids[i].insert(p_id);
                    geometry_msgs::Point32 p;
                    p.x = un_lines[j].norm_start.x;
                    p.y = un_lines[j].norm_start.y;
                    p.z = 1;

                    feature_lines->points.push_back(p);
                    id_of_line.values.push_back(p_id * NUM_OF_CAM + i);
                    // std::cout<< "feature tracking id: " <<p_id * NUM_OF_CAM + i<<" "<<p_id<<"\n";
                    u_of_endpoint.values.push_back(un_lines[j].norm_end.x);
                    v_of_endpoint.values.push_back(un_lines[j].norm_end.y);
                    // ROS_ASSERT(inBorder(cur_pts[j]));
                }
            }
        }
        feature_lines->channels.push_back(id_of_line);
        feature_lines->channels.push_back(u_of_endpoint);
        feature_lines->channels.push_back(v_of_endpoint);
        ROS_DEBUG("publish %f, at %f", feature_lines->header.stamp.toSec(), ros::Time::now().toSec());
        // if (!init_pub)
        // {
        //     init_pub = 1;
        // }
        // else
            pub_img.publish(feature_lines);

        {
            cv_bridge::CvImageConstPtr ptr1 = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::BGR8);
            cv::Mat tmp_img = ptr1->image;
            cv::cvtColor(trackerData.cur_img->img, tmp_img, CV_GRAY2RGB);

            const vector<Line> &un_lines = trackerData.cur_img->vecLine;
            for (unsigned int j = 0; j < trackerData.cur_img->vecLine.size(); j++)
            {
                if (trackerData.cur_img->success[j] != -1)
                    continue;
                ;

                Point2f start = un_lines[j].keyPoint[1];
                Point2f end = un_lines[j].keyPoint[un_lines[j].keyPoint.size() - 1];
                Point2f start1 = un_lines[j].StartPt;
                Point2f end1 = un_lines[j].EndPt;

                cv::line(tmp_img, start, end, cv::Scalar(0, 255, 0), 2, LINE_AA);
                cv::line(tmp_img, start1, end1, cv::Scalar(0, 255, 0), 1, LINE_AA);
                cv::putText(tmp_img, std::to_string(trackerData.cur_img->lineID[j]), trackerData.cur_img->vecLine[j].keyPoint[0], cv::FONT_HERSHEY_SIMPLEX, 0.45, CV_RGB(255, 230, 0), 1.8);
            }
            for (unsigned int m = 0; m < trackerData.cur_img->vecLine.size(); m++)
            {
                if (trackerData.cur_img->success[m] == -1)
                    break;

                Point2f start = un_lines[m].keyPoint[1];
                Point2f end = un_lines[m].keyPoint[un_lines[m].keyPoint.size() - 1];
                Point2f start1 = un_lines[m].StartPt;
                Point2f end1 = un_lines[m].EndPt;
                cv::line(tmp_img, start, end, cv::Scalar(255, 0, 0), 2, LINE_AA);
                cv::line(tmp_img, start1, end1, cv::Scalar(255, 0, 0), 1, LINE_AA);
                cv::putText(tmp_img, std::to_string(trackerData.cur_img->lineID[m]), trackerData.cur_img->vecLine[m].keyPoint[0], cv::FONT_HERSHEY_SIMPLEX, 0.45, CV_RGB(255, 230, 0), 1.8);
            }
            pub_match.publish(ptr1->toImageMsg());
        }
        {
#if _SHOWLINEFLOW_
            cv_bridge::CvImageConstPtr ptr1 = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::BGR8);
            cv::Mat detect_img = ptr1->image;
            cv::cvtColor(trackerData.cur_img->img, detect_img, CV_GRAY2RGB);
            for (unsigned int m = 0; m < trackerData.cur_img->lineRec.size(); m++)
                for (unsigned int j = 0; j < trackerData.cur_img->lineRec[m].size(); j++)
                {
                    Point2f start = trackerData.cur_img->lineRec[m][j].start;
                    Point2f end = trackerData.cur_img->lineRec[m][j].end;
                    if (j == trackerData.cur_img->lineRec[m].size() - 1)
                        cv::line(detect_img, start, end, cv::Scalar(0, 255, 0), 2, LINE_AA);
                    else
                        cv::line(detect_img, start, end, cv::Scalar(0, 0, 255), 1, LINE_AA);
                }
            pub_detect.publish(ptr1->toImageMsg());
#endif
        }
    }

    sum_time += t_r.toc();
    mean_time = sum_time / frame_cnt;
    // ROS_INFO("whole Line feature tracker processing costs: %f", mean_time);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "lf_linefeature_tracker");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    readParameters(n);

    for (int i = 0; i < NUM_OF_CAM; i++)
        trackerData.readIntrinsicParameter(CAM_NAMES[i]);

    ROS_INFO("start line feature");
    ros::Subscriber sub_img = n.subscribe(IMAGE_TOPIC, 100, img_callback);

    pub_img = n.advertise<sensor_msgs::PointCloud>("linefeature", 1000);
    pub_match = n.advertise<sensor_msgs::Image>("linefeature_img", 1000);
    pub_detect = n.advertise<sensor_msgs::Image>("linefeature_detect", 1000);

    /*
    if (SHOW_TRACK)
        cv::namedWindow("vis", cv::WINDOW_NORMAL);
    */
    ros::spin();
    return 0;
}
