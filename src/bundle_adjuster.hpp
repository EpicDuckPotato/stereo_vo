#ifndef BUNDLE_ADJUSTER_H_
#define BUNDLE_ADJUSTER_H_

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <cstdlib>
#include "camera_info.hpp"

#include "gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h"
#include "gtsam/nonlinear/NonlinearFactorGraph.h"
#include "gtsam/nonlinear/Values.h"
#include "gtsam/inference/Key.h"
#include "gtsam/geometry/Pose3.h"
#include "gtsam/geometry/Cal3Unified.h"

using namespace std;
using namespace Eigen;
using namespace gtsam;

/*
 * Keyframe: stores the position and orientation of the world frame
 * with respect to the camera frame for a keyframe, as well as the 
 * keyframe's associated image, 2d features, 3d features, and
 * feature ids, which are used to identify features as they
 * are tracked across multiple frames.
 * TODO: this structure is obsolete, and is only used to pass
 * things to the bundle adjuster once. Might as well just
 * pass all these things as individual arguments
 */
struct Keyframe {
  double timestamp;
  Vector3f position; 
  Quaternionf orientation; 
  cv::Mat image; 
  vector<cv::Point2f> tracked_features_2d; 
  vector<Key> tracked_keys;
  vector<cv::Point2f> new_features_2d; 
  vector<cv::Point3f> new_features_3d;
  vector<Key> new_keys; 

  Keyframe(double timestamp,
           Vector3f position, 
           Quaternionf orientation,
           cv::Mat image,
           vector<cv::Point2f> tracked_features_2d,
           vector<size_t> tracked_keys,
           vector<cv::Point2f> new_features_2d,
           vector<cv::Point3f> new_features_3d) : 
           timestamp(timestamp),
           position(position),
           orientation(orientation),
           image(image),
           tracked_features_2d(tracked_features_2d),
           tracked_keys(tracked_keys),
           new_features_2d(new_features_2d),
           new_features_3d(new_features_3d) {}
};

// Limit to number of features in a keyframe
static const size_t max_features = 400;

class BundleAdjuster
{
  public:
    /*
     * BundleAdjuster: constructor
     * ARGUMENTS
     * lag: time (in seconds) that we optimize over for fixed-lag smoothing
     * info: camera intrinsics
     */
    BundleAdjuster(double time_lag, CameraInfo info);

    /*
     * add_keyframe: adds a keyframe to the bundle adjustment window, and removes an old
     * keyframe if there are too many. Gives keys to new features, and throws out new
     * features that push us over the feature limit
     * ARGUMENTS
     * keyframe: keyframe to add
     */
    void add_keyframe(Keyframe &keyframe);

    /*
     * bundle_adjust: performs optimization over all keyframes in the window.
     */
    void bundle_adjust();

    /*
     * get_last_pose: gets the most recent estimated pose of the camera with
     * respect to the world
     * ARGUMENTS
     * position: populated with position
     * orientation: populated with orientation
     */
    void get_last_pose(Vector3d &position, Quaterniond &orientation);

    /*
     * get_world_points: gets world points associated with the given keys
     * ARGUMENTS
     * world_points: populated with world points
     * keys: keys of the world points to get
     */
    void get_world_points(vector<cv::Point3f> &world_points, const vector<Key> &keys);

    /*
     * any_keyframes: tells us whether we've added any keyframes to the optimization
     * RETURN true if we've added keyframes, false otherwise
     */
    inline bool any_keyframes() {
      return any_frames;
    }

  private:
    NonlinearFactorGraph newFactors;
    Values newValues;
    FixedLagSmoother::KeyTimestampMap newTimestamps;
    boost::shared_ptr<Cal3Unified> K;

    Key next_key;
    Key pose_key; // Key associated with the pose of the most recent keyframe

    IncrementalFixedLagSmoother smoother;

    bool new_frame_added;
    bool any_frames;
    bool optimized;
};

#endif
