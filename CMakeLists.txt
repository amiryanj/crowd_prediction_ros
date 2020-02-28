cmake_minimum_required(VERSION 2.8.3)
project(crowd_prediction)

find_package(catkin REQUIRED COMPONENTS
  std_msgs
  geometry_msgs
  message_generation
)

catkin_python_setup()


## Generate messages in the 'msg' folder
add_message_files(
  FILES
  Vec3Trajectory.msg
  Vec3GroupTrajectory.msg
  Vec3GroupTrajectoryArray.msg
  MovingParticleStamped.msg
  ParticleCrowd.msg
  ParticleCrowdArray.msg
)

## Generate services in the 'srv' folder
add_service_files(
  FILES
  GetPredictedTrajectories.srv
  GetSynthesizedCrowd.srv
)


## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)


catkin_package(
  CATKIN_DEPENDS rospy std_msgs geometry_msgs message_runtime 
)


## Build ##

include_directories(
 ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
			scripts/prediction_node.py
			scripts/trajec_prediction_client.py
			scripts/trajec_prediction_server.py
			scripts/trajec_prediction_server.py
		      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

