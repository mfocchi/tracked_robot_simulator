cmake_minimum_required(VERSION 3.8)
project(optim_interfaces)

find_package(catkin REQUIRED  message_generation)

add_service_files(
  DIRECTORY
  srv
  FILES
  Optim.srv
) 

generate_messages()

catkin_package()
