cmake_minimum_required(VERSION 2.8.3)
project(crowd_synthesis)

find_package(catkin REQUIRED COMPONENTS
  std_msgs
  geometry_msgs
  message_generation
)


catkin_python_setup()


## Generate messages in the 'msg' folder
add_message_files(
  FILES
  MovingParticleStamped.msg
  ParticleCrowd.msg
  ParticleCrowdArray.msg
)

## Generate services in the 'srv' folder
add_service_files(
  FILES
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

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
# ${catkin_INCLUDE_DIRS}
)

#############
## Install ##
#############
## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

