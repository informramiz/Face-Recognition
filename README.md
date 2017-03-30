# Face-Recognition

Face Recognition algorithm written in C++ using OpenCV and LBP classifier, for Windows.

## Getting Started

```
//set the paths to required files
string face_cascade_name = "\\lbpcascade_frontalface.xml";
string training_file_path = "\\lbp-live.yml";

//initialize Face Recognizer with right file paths
RFaceRecognizer face_recognizer (face_cascade_name, training_file_path);

//initialize MS DirectShow VideoCapture library
VideoCapture cap(0);

//label of subject
int label;

bool status = false;

//let's train recoginition model from live webcam video
status = face_recognizer.learn(cap, label);

//....
//....

//now if you have some more images of already trained subject then 
//update existing model
status = face_recognizer.Update(cap, label);

//let's recognize some subject from live webcam video
face_recognizer.recognize(cap , true );
```

There is also an installer .exe file that you can install on your system if just want to try it.


## Dependencies 

- It uses directshow so directshow paths must be set.
- It uses OpenCV so OpenCV must be configured.

## Possible Issues and Improvements

- I worked on it few years back so it is not up to date so it can definitely be improved and upgraded.
- At the time it was tested on Windows 7 and Windows 8 only.



