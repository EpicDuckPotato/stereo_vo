#include <bundle_adjuster.hpp>
#include <unordered_map>

/*
 * Tracks features from a given keyframe
 */
class FeatureTracker {
  public:
    /*
     * init: initializes the tracker with a set of features
     *       to track
     * ARGUMENTS
     * image: image containing features
     * features: features to track
     * ids: ids points corresponding to features to track
     */
    void init(const cv::Mat &image, const vector<cv::Point2f> &features,
              const vector<size_t> &ids);

    /*
     * track_features: finds the coordinates of the feature set being tracked
     *                 in the given image
     * ARGUMENTS
     * av_parallax: populated with average parallax of features from their initial image
     * percent_lost: populated with total percentage of features lost from the initial image
     * image: image into which we track the current feature set
     * flow_back: should we calculate optical flow in reverse as well? Potentially increases
     *            tracking accuracy in return for increased computation           
     */
    void track_features(float &av_parallax, float &percent_lost, const cv::Mat &image,
                        bool flow_back);

    /* 
     * get_tracked_features: gets the current 2d positions of the tracked feature
     *                       set and their associated ids
     * ARGUMENTS
     * features: populated with 2d positions of current feature set
     * ids: populated with ids of current feature set
     */
    void get_tracked_features(vector<cv::Point2f> &features, vector<size_t> &ids);

    /*
     * draw_track: draws vectors at each feature location in the initial image, pointing
     *             to their location in the most recent image
     */
    void draw_track();

    /*
     * get_drawing: gets the most recent drawing created with draw_track
     */
    cv::Mat get_drawing();

  private:
    cv::Mat initial_image;
    cv::Mat track_drawing;
    cv::Mat last_image;
    unordered_map<size_t, cv::Point2f> initial_features;
    vector<cv::Point2f> feature_set;
    vector<size_t> feature_ids;
};
