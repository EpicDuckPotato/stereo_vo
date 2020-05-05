#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <stack>
#include "camera_info.hpp"

using namespace std;
using namespace Eigen;

struct Keyframe {
  Vector3f position; 
  AngleAxisf orientation; 
  cv::Mat image; 
  vector<cv::Point2f> features_2d; 
  vector<cv::Point3f> features_3d;
  vector<size_t> feature_ids;

  Keyframe(Vector3f position, 
           AngleAxisf orientation,
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

// A variable representing a camera pose in the factor graph 
struct PoseVariable {
  double *position; 
  double *orientation; // rotation vector

  // When we pop this from the window, we need to know which residual blocks to
  // remove from the Ceres problem. We also need to know which features those
  // blocks correspond to, to decrement the refcount
  vector<pair<ceres::ResidualBlockId, size_t>> observations;

  PoseVariable() {
    position = new double[3];
    orientation = new double[3];
  }
};

struct Feature {
  double *position; // 3d position in world frame

  // Every time this feature is observed in a keyframe, we increment the
  // refcount. Every time a pose is popped from the bundle adjustment window,
  // we decrement the refcount. In decrementing, if we find that the refcount
  // becomes zero, we push this feature's index onto the empty_features stack,
  // since this feature isn't being observed by any keyframes in the window.
  size_t refcount; 
};

// Limit to number of features in a keyframe
static const size_t max_features = 200;

class BundleAdjuster
{
  public:
    BundleAdjuster(size_t _window_size, CameraInfo info);

    inline shared_ptr<Keyframe> get_last_keyframe() {
      return last_keyframe;
    }

    /*
     * add_keyframe: assumes that the observations in keyframe->features_2d correspond to the
     * same features as get_last_keyframe()->features_2d. If there are more observations in
     * keyframe->features_2d than get_last_keyframe()->features_2d, these are added to the feature list,
     * provided there's room. Any features that don't fit in the feature list are discarded.
     * After this function returns, calls to get_last_keyframe() will return a
     * pointer to this keyframe.
     */
    void add_keyframe(shared_ptr<Keyframe> keyframe);

    void bundle_adjust();
  private:
    void removeOldestPose();

    vector<Feature> features; 

    // available feature ids, i.e. indices of empty elements of the list
    stack<size_t> avail_ids; 

    queue<shared_ptr<PoseVariable>> pose_window; // window of poses to optimize over
    size_t window_size; // max poses in window

    shared_ptr<Keyframe> last_keyframe;

    ceres::Problem problem;
    ceres::Solver::Options options;

    CameraInfo camera_info;
};
