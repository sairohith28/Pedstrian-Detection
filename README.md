# Pedestrian Detection and Tracking
Image name:newimage
Image id :ed382d4acb05

**Pedestrian Detection Docker Containerization:**

- First of all start the docker desktop application and make sure that it is running in background.

- Get the Docker file from[here](/uploads/996d1d102da755979b13260f0ef55c25/dockerfile)

- Make Sure that working folder structure should be like this 

       Working_Folder
       
       |_Demo
       |_Model
       |_Dockerfile

- After navigating into the folder run the following command to create docker image 

       docker build -t newimage .

- After successful creation of docker container just start the container using following command 

       docker run -ti newimage /bin/bash

- Now create a folder named output to store the output images.

- Navigate to the demo folder where Inference file exist and run the inference file using following commanad 

       python3 Inference.py

- After a successful run of inference file the output images are stored in output folder. it shows like this 

   ![Screenshot_20230218_005730](/uploads/9af98338e11f4c0df43f1418335ac3a1/Screenshot_20230218_005730.png)

- Now to copy the output images from Container to host machine use the following command and give file paths accordingly 

       docker cp {container_id}:output {path_in_host}

- Now the output images will be stored in host machine.

