### 30/06/2026
Given the poor results from vgicp (vgicp/*) and the acceptable results from kiss_matcher (kiss_matcher/b5315e5114194b3aa60109ee99936bac), I'll try another global method and then use those two (kiss_matcher and new) as initialization for finer registration methods.

### 06/07/2026
Given the results from the identity (identity/*) and vgicp(vgicp-128-imu), it's clear V2X-Real initialization with lidar_pose is not the same as an IMU-based initialization, since the lidar pose is basically the ground truth unlike previously thought. To try an IMU-based initialization approach another dataset should be used.