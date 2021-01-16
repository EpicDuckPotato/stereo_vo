#include "bundle_adjuster.hpp"
#include "feature_tracker.hpp"
#include <opencv2/opencv.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/core/eigen.hpp"
#include <Eigen/Dense>


struct StereoPair {
  cv::Mat left;
  cv::Mat right;
  double t;

  StereoPair(const cv::Mat &left, const cv::Mat &right, double t) : left(left),
                                                                    right(right),
                                                                    t(t) {}
};

class ImageProcessor {
  public:
    /*
     * ImageProcessor: constructor
     * ARGUMENTS
     * cam_mat: camera_matrix;
     * tracker: FeatureTracker to track features
     * adjuster: BundleAdjuster object to add keyframes to 
     * bline: stereo baseline
     * min_feature_distance: min distance between detected features
     * parallax_thresh: min average parallax for us to care about an image
     */
    ImageProcessor(cv::Mat cam_mat, 
                   shared_ptr<FeatureTracker> tracker,
                   shared_ptr<BundleAdjuster> adjuster,
                   float bline,
                   float min_feature_distance,
                   float parallax_thresh);

    /*
     * process: we use image pairs for processing only if features
     * from the last keyframe have sufficient parallax with this pair's left image.
     * If so, use previous keyframe's 3d feature positions to solve for current left
     * camera pose via PnP
     * ARGUMENTS
     * stereo_pair: pair of stereo images and timestamp
     */
    void process(const StereoPair &stereo_pair);

  private:
    shared_ptr<BundleAdjuster> bundle_adjuster;
    shared_ptr<FeatureTracker> feature_tracker;
    cv::Mat camera_matrix;
    float baseline;

    float min_feature_distance;
    float parallax_thresh;

    // We use the previous transform (i.e. the previous solvePnP output) as the
    // initial guess for the next PnP solution
    cv::Mat rvec;
    cv::Mat tvec;

    /*
     * triangulate_stereo: triangulates the 3d positions of 2d features
     * ARGUMENTS
     * features_3d: populated with the 3d positions of valid_features_2d
     * valid_features_2d: populated with elements of left_features_2d with nonzero disparity
     * the left and right images, which is a condition necessary for
     * triangulation. This vector is populated with all
     * 2d feature positions that were able to take part in triangulation
     * left_features_2d: 2d positions in left image to triangulate
     * left: left image
     * right: right image
     * camera_pose: 4x4 matrix transforming points in camera frame to
     *              world frame
     * REQUIRES: features_3d.size() == 0 
     */
    void triangulate_stereo(vector<cv::Point3f>& features_3d,
                            vector<cv::Point2f>& valid_features_2d,
                            const vector<cv::Point2f>& left_features_2d,
                            const cv::Mat& left,
                            const cv::Mat& right,
                            const cv::Mat& camera_pose);
};
