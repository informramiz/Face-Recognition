#ifndef RFaceRecognizer_H_
#define RFaceRecognizer_H_

#include <iostream>
#include <vector>
#include <string>
#include <cv.h>
#include <cor.h>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\contrib\contrib.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <highgui.h>

using namespace std;
using namespace cv;

class RFaceRecognizer
{

public:
	RFaceRecognizer( std::string & face_cascade_file_path , std::string & training_file_path );
	bool IsTrained( );
	bool RFaceRecognizer::recognize( cv::VideoCapture & cap , bool recognize_multiple_faces );

	bool learn ( VideoCapture & cap , int label = 1  );
	bool Update ( VideoCapture & cap , int label = 1  );
private:
	void LoadData( );

private:
	string num2str ( int value );
	int str2num ( const string & value );
	bool FaceDetector ( const Mat & image , Mat & dest );
	bool FaceDetector ( const Mat & image , std::vector<cv::Rect> & faces );
	bool FaceDetector ( const Mat & image , Mat & dest, Rect & faceReturn );

	bool readDataSet ( const string & dataSetPath , std::vector<cv::Mat> & images_vector , std::vector<int> & labels_vector );
	bool readDataSet ( cv::VideoCapture & cap , std::vector<Mat> & images_vector , std::vector<int> & labels_vector, int lablel );
	bool eyeTranslator ( Mat & face , const Rect & leftEye , const Rect & rightEye );
	bool eyeDetector ( const Mat & face , Rect & eye , string eyeFile );

	void smooth ( Mat & face );
	bool learn ( const std::string & dataSetPath , const std::string & outputFilePath );

	bool recognize ( const cv::Mat & testImage , int & predictedLabel , double & confidence , cv::Rect & fRect);
	bool RFaceRecognizer::recognize ( const cv::Mat & testImage , vector<Rect> & faces, vector<int> & labels, vector<double> & confidence );
	bool recognize( const std::string & testImagePath , int & predictedLabel , double & confidence );
	bool recognizeMultipleFaces( cv::VideoCapture & cap );
	bool recognizeSingleFace( cv::VideoCapture & cap );
private:
	static const double EYE_SX ;
	static const double EYE_SY ;
	static const double EYE_SW;
	static const double EYE_SH;

	static const int DESIRED_IMAGE_WIDTH;
	static const int DESIRED_IMAGE_HEIGHT;
	static const int DESIRED_FACE_WIDTH;
	static const int DESIRED_FACE_HEIGHT;

	static const int noOfSubjects;
	static const int noOfImages;
	static const int testCol;

	static const string leftEyeFile ;
	static const string rightEyeFile ;

private:
	std::string face_cascade_file_path_;
	std::string training_file_path_;

	cv::CascadeClassifier face_cascade_classifier_;
	cv::Ptr<cv::FaceRecognizer> face_recognizer_;

	bool is_trained_;
};

#endif