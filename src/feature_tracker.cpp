#include "feature_tracker.hpp"

void FeatureTracker::init(const cv::Mat &image, const vector<cv::Point2f> &features,
                          const vector<size_t> &ids) {
  feature_ids = ids;
  feature_set = features;

  size_t num_features = ids.size();
  initial_features.clear();
  for (size_t i = 0; i < num_features; i++) { 
    initial_features.insert(make_pair(ids[i], features[i])); 
  }

  last_image = image.clone();
  initial_image = last_image;
}

void FeatureTracker::track_features(float &av_parallax, float &percent_lost, const cv::Mat &image) {
  vector<uchar> status;
  vector<float> err;
  vector<cv::Point2f> tracked_features;
  cv::calcOpticalFlowPyrLK(last_image, image, feature_set, tracked_features, status,
                           err, cv::Size(21, 21), 3,
                           cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01),
                           0, 1e-4);

  feature_set.clear();
  feature_ids.clear();
  av_parallax = 0;
  size_t num_features = tracked_features.size();
  for (size_t i = 0; i < num_features; i++) {
    if (status[i]) {
      feature_set.push_back(tracked_features[i]);
      feature_ids.push_back(feature_ids[i]);

      float dx = tracked_features[i].x - initial_features.at(feature_ids[i]).x;
      float dy = tracked_features[i].y - initial_features.at(feature_ids[i]).y;

      av_parallax += sqrt(dx*dx + dy*dy);
    }
  }

  av_parallax /= (float)num_features;
  percent_lost = 1.0 - feature_set.size()/(float)(initial_features.size());

  last_image = image.clone();
}

void FeatureTracker::get_tracked_features(vector<cv::Point2f> &features, vector<size_t> &ids) {
  features = feature_set;
  ids = feature_ids;
}

void FeatureTracker::draw_track(cv::Mat &track) {
  track = initial_image.clone();
  size_t num_features = feature_set.size();
  for (size_t i = 0; i < num_features; i++) {
    cv::Point2f direction = feature_set[i] - initial_features.at(feature_ids[i]);
    cv::arrowedLine(track, initial_features.at(feature_ids[i]),
                    initial_features.at(feature_ids[i]) + direction, CV_RGB(255, 255, 255), 4);
  }
}
