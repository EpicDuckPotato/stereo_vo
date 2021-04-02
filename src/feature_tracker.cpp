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

void FeatureTracker::track_features(float &av_parallax, float &percent_lost, const cv::Mat &image,
                                    bool flow_back) {
  vector<uchar> status1;
  vector<float> err;
  vector<cv::Point2f> tracked_features;
  cv::calcOpticalFlowPyrLK(last_image, image, feature_set, tracked_features, status1,
                           err, cv::Size(21, 21), 3,
                           cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01),
                           0, 1e-2);

  vector<uchar> status2;
  vector<cv::Point2f> reverse_track;
  
  if (flow_back) {
    cv::calcOpticalFlowPyrLK(image, last_image, tracked_features, reverse_track, status2,
                             err, cv::Size(21, 21), 3,
                             cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01),
                             0, 1e-2);
  }

  vector<cv::Point2f> old_feature_set(feature_set);
  vector<size_t> old_feature_ids(feature_ids);
  feature_set.clear();
  feature_ids.clear();
  av_parallax = 0;
  size_t num_features = tracked_features.size();
  for (size_t i = 0; i < num_features; i++) {
    if (status1[i] && 
        (!flow_back || 
         (status2[i] && cv::norm(old_feature_set[i] - reverse_track[i]) < 2))) {
      float dx = tracked_features[i].x - initial_features.at(old_feature_ids[i]).x;
      float dy = tracked_features[i].y - initial_features.at(old_feature_ids[i]).y;

      float parallax = sqrt(dx*dx + dy*dy);
  
      if (parallax > 200) {
        continue;
      }

      feature_set.push_back(tracked_features[i]);
      feature_ids.push_back(feature_ids[i]);
      av_parallax += parallax;
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

void FeatureTracker::draw_track() {
  track_drawing = initial_image.clone();
  cv::cvtColor(track_drawing, track_drawing, CV_GRAY2RGB);
  size_t num_features = feature_set.size();
  for (size_t i = 0; i < num_features; i++) {
    cv::Point2f direction = feature_set[i] - initial_features.at(feature_ids[i]);
    cv::arrowedLine(track_drawing, initial_features.at(feature_ids[i]),
                    initial_features.at(feature_ids[i]) + direction, CV_RGB(0, 255, 0), 4);
  }
}

cv::Mat FeatureTracker::get_drawing() {
  if (track_drawing.empty()) {
    track_drawing = initial_image.clone();
    cv::cvtColor(track_drawing, track_drawing, CV_GRAY2RGB);
  }
  return track_drawing;
}
