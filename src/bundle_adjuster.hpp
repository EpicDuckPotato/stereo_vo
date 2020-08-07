#ifndef BUNDLE_ADJUSTER_H_
#define BUNDLE_ADJUSTER_H_

#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <stack>
#include "camera_info.hpp"
#include "ceres/local_parameterization.h"

using namespace std;
using namespace Eigen;

/*
 * Keyframe: stores the position and orientation of the world frame
 * with respect to the camera frame for a keyframe, as well as the 
 * keyframe's associated image, 2d features, 3d features (whose indices
 * correspond with their 2d counterparts' in features_2d), and
 * feature ids, which are used to identify features as they
 * are tracked across multiple frames
 */
struct Keyframe {
  Vector3f position; 
  Quaternionf orientation; 
  cv::Mat image; 
  vector<cv::Point2f> features_2d; 
  vector<cv::Point3f> features_3d;
  vector<size_t> feature_ids;

  Keyframe(Vector3f position, 
           Quaternionf orientation,
           cv::Mat image,
           vector<cv::Point2f> features_2d,
           vector<cv::Point3f> features_3d,
           vector<size_t> feature_ids) : 
           position(position),
           orientation(orientation),
           image(image),
           features_2d(features_2d),
           features_3d(features_3d),
           feature_ids(feature_ids) {}
};

// A variable representing a keyframe's camera pose in the factor graph 
struct PoseVariable {
  double *pose; // wxyz (orientation), xyz (position)

  // When we pop this from the window, we need to know which residual blocks to
  // remove from the Ceres problem. We also need to know which feature ids those
  // blocks correspond to, to decrement the refcount
  vector<pair<ceres::ResidualBlockId, size_t>> observations;

  PoseVariable() {
    pose = new double[7];
  }
};

struct Feature {
  double *position; // 3d position in world frame

  // Every time this feature is observed in a keyframe, we increment the
  // refcount. Every time a keyframe is popped from the bundle adjustment window,
  // and this feature was observed at that keyframe, we decrement the refcount.
  // In decrementing, if we find that the refcount becomes zero,
  // we push this feature's index onto the avail_ids stack,
  // since this feature isn't being observed by any keyframes in the window
  size_t refcount; 
};

// Limit to number of features in a keyframe
static const size_t max_features = 400;

class BundleAdjuster
{
  public:
    /*
     * BundleAdjuster: constructor
     * ARGUMENTS
     * info: camera intrinsics
     * _window_size: number of keyframes to optimize in the sliding window
     */
    BundleAdjuster(size_t _window_size, CameraInfo info);

    /*
     * ~BundleAdjuster: destructor
     */
    ~BundleAdjuster();

    /*
     * get_last_keyframe: return a pointer to the last keyframe passed
     * to this object with add_keyframe
     * RETURN: a pointer to the last keyframe passed
     * to this object with add_keyframe
     */
    inline shared_ptr<Keyframe> get_last_keyframe() {
      return last_keyframe;
    }

    /*
     * add_keyframe: adds a keyframe to the bundle adjustment window, and removes an old
     * keyframe if there are too many. If there are more observations in
     * keyframe->features_2d than get_last_keyframe()->features_2d, these are added to the feature list.
     * Any features whose indices exceed max_features are discarded
     * After this function returns, calls to get_last_keyframe() will return a
     * pointer to this keyframe.
     * ARGUMENTS
     * keyframe: pointer to the keyframe to add
     */
    void add_keyframe(shared_ptr<Keyframe> keyframe);

    /*
     * bundle_adjust: performs optimization over all keyframes in the window and
     * updates the position and orientation of the last keyframe. Any subsequent calls
     * to get_last_keyframe will reflect the optimization
     */
    void bundle_adjust();

    /*
     * get_world_points: gets world points associated with the given ids
     * ARGUMENTS
     * world_points: populated with world points
     * ids: ids of the world points to get
     */
    void get_world_points(vector<cv::Point3f> &world_points, const vector<size_t> &ids);
  private:
    /*
     * remove_oldest_pose: removes the oldest pose from the bundle adjustment window
     */
    void remove_oldest_pose();

    // list of all features that are observed by some keyframe in the window
    vector<Feature> features; 

    // available feature ids, i.e. indices of empty elements of the list
    stack<size_t> avail_ids; 

    queue<shared_ptr<PoseVariable>> pose_window; // window of poses to optimize over
    size_t window_size; // max poses in window

    shared_ptr<Keyframe> last_keyframe;

    unique_ptr<ceres::Problem> problem;
    ceres::Solver::Options options;
    unique_ptr<ceres::ProductParameterization> se3param;

    CameraInfo camera_info;
    bool new_frame_added;
};

#endif
